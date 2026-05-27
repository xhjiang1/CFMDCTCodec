from abc import ABC
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fm_decoder import Decoder
from net_utils import get_pylogger

log = get_pylogger(__name__)

def build_sigma_from_mdct_energy(
    X: torch.Tensor,
    min_sigma: float = 1e-3,
    max_sigma: float = 1.0,
    time_pool: int = 3,
    freq_pool: int = 5,
    q: float = 0.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    根据 MDCT (或频域特征) 的能量自适应生成逐 bin 的 σ。
    目标：静音/弱能量区 σ 小、强能量区 σ 大，减少“运输距离”。

    支持输入排布：
      (B, C, T, F) 或 (B, T, F) 或 (B, F, T)

    返回：与 X 同形状的 sigma，用于 sigma * torch.randn_like(X)
    """
    x = X
    # -------- 统一到 (B, C, T, F) --------
    if x.dim() == 3:
        B, A, Bdim = x.shape
        # 猜测 (T,F) 排布：默认 A=T, Bdim=F
        # 若你的数据实际是 (B,F,T)，改成 x = x.permute(0,2,1)
        x = x.unsqueeze(1)           # (B,1,T,F)
    elif x.dim() == 4:
        pass                         # (B,C,T,F)
    else:
        raise ValueError("Expect X with 3 or 4 dims")

    B, C, T, Freq = x.shape

    # -------- 能量代理：支持复数，取幅值；通道聚合减少抖动 --------
    mag = x.abs()                    # (B,C,T,F)
    mag = mag.mean(dim=1, keepdim=True)  # (B,1,T,F)

    # -------- 时频平滑（避免 σ 尖锐）--------
    mag = F.avg_pool2d(
        mag, kernel_size=(time_pool, freq_pool),
        stride=1, padding=(time_pool//2, freq_pool//2)
    )                                # (B,1,T,F)

    # -------- 幅度标度 + 分位数归一化（每个样本单独）--------
    amp = torch.sqrt(mag + eps)      # (B,1,T,F)
    # 展平到 (B, 1, T*F) 做分位数，避免异常峰值主导
    denom = torch.quantile(
        amp.flatten(start_dim=2), q, dim=2, keepdim=True
    ).clamp_min(eps)                 # (B,1,1)
    denom = denom.unsqueeze(-1)      # (B,1,1,1)
    sigma = (amp / denom).clamp(min_sigma, max_sigma)  # (B,1,T,F)

    # -------- 广播回通道维 --------
    sigma = sigma.expand(-1, C, -1, -1)  # (B,C,T,F)

    # -------- 还原到输入排布 --------
    if X.dim() == 3:
        sigma = sigma[:, 0]         # (B,T,F)
    # 否则保持 (B,C,T,F)

    return sigma

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        cfm_params = SimpleNamespace(**cfm_params)
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """
        Flow-matching 推理（MDCT 版）。
        从 decoder 输出 mu 的“条件化先验”附近出发，沿学到的速度场积分到 t=1。
        """
        # 1) 条件化 σ：与训练保持一致
        with torch.no_grad():
            sigma = build_sigma_from_mdct_energy(mu)      # 形状与 mu 一致，含 clamp

            # 2) 随机起点：mu + temperature * sigma * N(0, I)
            #    注意：temperature 可以是标量，也可以允许传入同形状张量（更灵活）
            noise = torch.randn_like(mu)
            x0 = mu + (temperature * sigma) * noise

            # 3) 时间网格
            # 若你训练时 t~U(0,1) 没有在 0.9~1.0 封顶加权，这里用 1.0 没问题；
            # 若你训练里对 t→1 做了截断（比如只到 0.99），推理也可一致到 0.99。
            t_end = 1.0
            t_span = torch.linspace(0.0, t_end, n_timesteps + 1,
                                    device=mu.device, dtype=mu.dtype)

        # 4) ODE 积分（概率流 ODE 的欧拉法）
        return self.solve_euler(x0, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)


    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed-step Euler ODE solver for JFM (real-valued MDCT).
        Integrates from t=0 -> 1 following dX/dt = v_theta(X_t, t, Y=mu).
        
        Args:
            x (Tensor): initial state (noisy mu)
            t_span (Tensor): 1D time grid (n_steps+1,)
            mu (Tensor): decoder output (condition)
            mask (Tensor): optional mask
            spks, cond: optional conditioning
        """
        dt = t_span[1] - t_span[0]
        t = t_span[0]

        for step in range(1, len(t_span)):
            # Predict instantaneous flow direction
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            # Euler integration
            x = x + dt * dphi_dt
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        # x is now the enhanced MDCT spectrum
        return x


    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """
        Computes Joint Flow Matching (JFM) loss for real-valued features (e.g. MDCT).

        Args:
            x1 (torch.Tensor): clean target MDCT features, shape (B, C, T)
            mask (torch.Tensor): valid time mask, shape (B, 1, T)
            mu (torch.Tensor): decoder output MDCT (conditional start, 'Y'), shape (B, C, T)
            spks (torch.Tensor, optional): speaker embedding
            cond (optional): optional conditioning

        Returns:
            loss (torch.Tensor): scalar JFM loss
            Xt (torch.Tensor): sampled intermediate point (for debugging)
        """
        B = mu.shape[0]

        # 1. Sample timestep t ~ U(0,1)
        t = torch.rand(B, device=mu.device, dtype=torch.float32).view(B, 1, 1)
        sigma = build_sigma_from_mdct_energy(mu.permute(0,2,1)).permute(0,2,1)
        # 2. Add endpoint noise (as in FlowDec Eq. 9)
        eps_y = torch.randn_like(mu)
        # eps_x = torch.randn_like(x1)
        Ys = mu + sigma * eps_y   # noisy start
        Xs = x1    # noisy target

        # 3. Interpolate intermediate point
        Xt = Ys + t * (Xs - Ys)          # linear path between endpoints

        # 4. Ground-truth flow field
        Ut = Xs - Ys                     # constant flow along the path

        # 5. Predicted flow
        Vt = self.estimator(Xt, mask, mu, t.squeeze(), spks, cond)

        # 6. Compute mean squared error (real-valued)
        squared_errs = (Vt - Ut) ** 2

        # Optional per-frequency weighting
        if getattr(self, "error_weighting", None) is not None:
            squared_errs = self.error_weighting.to(squared_errs.device) * squared_errs

        # 7. Reduce to per-sample errors
        per_sample_errs = squared_errs.flatten(start_dim=1).mean(dim=1)

        # 8. Handle potential NaNs
        isnans = torch.isnan(per_sample_errs)
        if torch.any(isnans):
            log.warning(f"NaNs detected in batch: {isnans}")
            per_sample_errs = per_sample_errs[~isnans]
        if torch.all(isnans):
            raise ValueError("Whole batch produced NaN loss — training likely unstable.")

        # 9. Final loss
        loss = torch.mean(per_sample_errs)
        return loss, Xt


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
        )

        in_channels = in_channels
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)

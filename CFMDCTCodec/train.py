import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import IMDCT, Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from models import Encoder, Decoder, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, amplitude_loss, phase_loss, STFT_consistency_loss, MultiResolutionDiscriminator
from utils import AttrDict, build_env, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from flowmaatching import CFM
import torch.nn as nn
torch.backends.cudnn.benchmark = True

class MDCTAmplitudeCompressor(nn.Module):
    """
    可逆的 MDCT 幅度压缩 + 标准化到 [-1, 1]。

    - 输入:  X  (实数 MDCT 系数, 形状 [B, T, H] 或 [B, H, T])
    - 输出:  Y, scale
        Y      : 压缩 + 标准化后的 MDCT，范围约 [-1, 1]
        scale  : 每个样本的缩放因子，用于 invert() 还原

    alpha < 1: 压缩动态范围
    """
    def __init__(self, alpha: float = 0.3, eps: float = 1e-8, time_freq_dims=(1, 2)):
        super().__init__()
        assert alpha > 0, "alpha 必须 > 0"
        self.alpha = alpha
        self.eps = eps
        self.time_freq_dims = time_freq_dims  # e.g. (1,2) for [B,T,H]

    def forward(self, X: torch.Tensor):
        """
        X: [B, T, H] 或 [B, H, T] 的实数 MDCT 谱
        返回: Y, scale
          Y     : 标准化到 [-1,1] 的压缩系数
          scale : 每个样本的缩放因子，用于 invert
        """
        # 保留符号，符号就是“相位”
        sign = torch.sign(X)
        mag = X.abs().clamp_min(self.eps)

        # 幂压缩: 大的变小，小的变大一点，动态范围压平
        if self.alpha != 1.0:
            mag_comp = mag.pow(self.alpha)
        else:
            mag_comp = mag

        # 每个样本自己的 max，保证 |Y| <= 1
        # 比如 X shape = [B, T, H]，那就对 T,H 两个维度求 max
        reduce_dims = self.time_freq_dims
        scale = mag_comp.amax(dim=reduce_dims, keepdim=True).clamp_min(self.eps)

        Y = sign * (mag_comp / scale)  # 现在 Y ∈ [-1, 1]

        return Y, scale

    def invert(self, Y: torch.Tensor, scale: torch.Tensor):
        """
        还原到压缩前的 MDCT 系数。
        需要 forward 时返回的 scale。
        """
        sign = torch.sign(Y)
        mag_comp = Y.abs() * scale  # 把标准化逆回去

        # 逆幂变换
        if self.alpha != 1.0:
            mag = mag_comp.clamp_min(self.eps).pow(1.0 / self.alpha)
        else:
            mag = mag_comp

        X_rec = sign * mag
        return X_rec


def train(h):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda:{:d}'.format(0))
    else:
        device = torch.device('cpu')
    IMDCT_operation=IMDCT(80).to(device)
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    n_timesteps = 6
    n_timesteps_train = 6
    temperature = 1.0
    n_feats = 40
    cfm_params = {
        "name": "CFM",
        "solver": "euler",
        "sigma_min": 1e-4
    }
    decoder_params = {
                "channels": [256, 256],
                "dropout": 0.05,
                "attention_head_dim": 64,
                "n_blocks": 2,
                "num_mid_blocks": 2,
                "num_heads": 2,
                "act_fn": "snakebeta"
    }

    print("Encoder: ")
    print(encoder)
    print("Decoder: ")
    print(decoder)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)
    cfmmodel = CFM(in_channels=n_feats*2, out_channel=n_feats, cfm_params=cfm_params, decoder_params=decoder_params).to(device)
    num_params = 0
    for p in cfmmodel.parameters():
        num_params += p.numel()
    print('Total Parameters: {:.3f}M'.format(num_params/1e6))    
    if os.path.isdir(h.checkpoint_path):
        cp_encoder = scan_checkpoint(h.checkpoint_path, 'encoder_')
        cp_decoder = scan_checkpoint(h.checkpoint_path, 'decoder_')
        # cp_do = scan_checkpoint(h.checkpoint_path, 'do_')
        cp_cfmmodel = scan_checkpoint(h.checkpoint_path, 'cfmmodel_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')
    steps = 0
    if cp_encoder is None or cp_decoder is None or cp_cfmmodel is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_cfmmodel = load_checkpoint(cp_cfmmodel, device)
        state_dict_encoder = load_checkpoint(cp_encoder, device)
        state_dict_decoder = load_checkpoint(cp_decoder, device)
        state_dict_do = load_checkpoint(cp_do, device)
        encoder.load_state_dict(state_dict_encoder['encoder'])
        decoder.load_state_dict(state_dict_decoder['decoder'])
        cfmmodel.load_state_dict(state_dict_cfmmodel['cfmmodel'])
        # mpd.load_state_dict(state_dict_do['mpd'])
        # mrd.load_state_dict(state_dict_do['mrd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    optim_g = torch.optim.AdamW(itertools.chain(encoder.parameters(), decoder.parameters(),cfmmodel.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    # optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        # optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                       h.hop_size, h.win_size, h.sampling_rate, h.ratio, n_cache_reuse=0,
                       shuffle=True, device=device)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    validset = Dataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                       h.hop_size, h.win_size, h.sampling_rate, h.ratio, False, False, n_cache_reuse=0,
                       device=device)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    encoder.train()
    decoder.train()
    cfmmodel.train()
    # mpd.train()
    # mrd.train()
    compressor = MDCTAmplitudeCompressor(alpha=0.5, time_freq_dims=(2, 1)).to(device)
    for epoch in range(max(0, last_epoch), h.training_epochs):

        start = time.time()
        print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            MDCT_coff, y, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            MDCT_coff = torch.autograd.Variable(MDCT_coff.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            latent,commitment_loss,codebook_loss = encoder(MDCT_coff)

            y_g, MDCT_g_coff = decoder(latent)
            MDCT_t_comp, _ = compressor(MDCT_coff)
            MDCT_g_comp, scale = compressor(MDCT_g_coff)
            mask = torch.ones(MDCT_g_coff.size(0), 1, MDCT_g_coff.size(-1)).to(device)

            loss_fm1, _ = cfmmodel.compute_loss(MDCT_t_comp, mask, MDCT_g_comp)
            
            # MDCT_hat_comp = cfmmodel(MDCT_g_comp, mask, n_timesteps_train, temperature)
            # MDCT_hat = compressor.invert(MDCT_hat_comp, scale)
            # y_g_fm = IMDCT_operation((MDCT_hat).permute(0,2,1)) 
            y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                      0, None)
            # y_g_fm_mel = mel_spectrogram(y_g_fm.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
            #                           0, None)
            
            optim_g.zero_grad()
            L_Mel = F.l1_loss(y_mel, y_g_mel)
            # L_Mel_fm = F.l1_loss(y_mel, y_g_fm_mel)
            L_Mel_L2 = amplitude_loss(y_mel, y_g_mel)
            # L_Mel_L2_fm = amplitude_loss(y_mel, y_g_fm_mel)         
            L_MDCT = amplitude_loss(MDCT_coff,MDCT_g_coff)
            # L_MDCT_fm = F.l1_loss(MDCT_hat_comp,MDCT_g_comp)



            L_W = 20 * L_Mel + 10 * L_Mel_L2 + 250 * L_MDCT
            L_G = L_W + codebook_loss*10 +commitment_loss*2.5 + loss_fm1 * 100
            L_G.backward()
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    Mel_error = F.l1_loss(y_mel, y_g_mel).item()
                    commit_loss = commitment_loss.item()
                    L_MDCT = F.l1_loss(MDCT_coff,MDCT_g_coff).item()
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, MDCT Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, FM Loss : {:4.3f}, Commit Loss : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, L_G, L_MDCT, Mel_error, loss_fm1.item(), commit_loss, time.time() - start_b))

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/cfmmodel_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'cfmmodel': cfmmodel.state_dict()})
                checkpoint_path = "{}/encoder_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'encoder': encoder.state_dict()})
                checkpoint_path = "{}/decoder_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'decoder': decoder.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, 
                                {
                                 'optim_g': optim_g.state_dict(),  'steps': steps,
                                 'epoch': epoch})

            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if steps % h.validation_interval == 0:
                encoder.eval()
                decoder.eval()
                cfmmodel.eval()
                torch.cuda.empty_cache()
                val_Mel_err_tot = 0
                val_Mel_L2_err_tot = 0
                val_MDCT_loss = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        MDCT_coff, y, y_mel = batch
                        latent,_,_ = encoder(MDCT_coff.to(device))
                        _,MDCT_g_coff = decoder(latent)
                        MDCT_g_coff=MDCT_g_coff.to(device)
                        MDCT_coff=MDCT_coff.to(device)

                        mask = torch.ones(MDCT_g_coff.size(0), 1, MDCT_g_coff.size(-1)).to(device)
                        MDCT_g_comp, scale_g = compressor(MDCT_g_coff)
                        MDCT_hat_comp = cfmmodel(MDCT_g_comp, mask, n_timesteps, temperature)
                        MDCT_hat = compressor.invert(MDCT_hat_comp, scale_g)
                        y_g = IMDCT_operation((MDCT_hat).permute(0,2,1)) 
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                        y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate,h.hop_size, h.win_size, 0, None)
                        
                        val_Mel_err_tot += F.l1_loss(y_mel, y_g_mel).item()
                        val_Mel_L2_err_tot += amplitude_loss(y_mel, y_g_mel).item()
                        val_MDCT_loss += amplitude_loss(MDCT_coff,MDCT_hat).item()
                        if j <= 4:
                            if steps == 0:
                                sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)

                            sw.add_audio('generated/y_g_{}'.format(j), y_g[0], steps, h.sampling_rate)
                        if j == 100:
                            break
                    val_Mel_err = val_Mel_err_tot / (j+1)
                    val_Mel_L2_err = val_Mel_L2_err_tot / (j+1)
                    val_MDCT_loss = val_MDCT_loss / (j+1)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_L2_loss", val_Mel_L2_err, steps)
                    sw.add_scalar("Validation/val_MDCT_loss", val_MDCT_loss, steps)

                encoder.train()
                decoder.train()
                cfmmodel.train()
            steps += 1

        scheduler_g.step()
        # scheduler_d.step()
        
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, "r", encoding="utf-8") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, 'config.json', h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == '__main__':
    main()

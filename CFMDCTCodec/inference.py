from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import librosa
import soundfile as sf
import torch

from dataset import IMDCT, MDCT
from flowmaatching import CFM
from models import Decoder, Encoder
from train import MDCTAmplitudeCompressor
from utils import AttrDict


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def load_checkpoint(filepath, device):
    if not filepath:
        raise ValueError("Checkpoint path is empty.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError("Checkpoint not found: {}".format(filepath))
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def collect_audio_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError("Input directory does not exist: {}".format(root))
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files)


def build_cfm_model(device):
    n_feats = 40
    cfm_params = {
        "name": "CFM",
        "solver": "euler",
        "sigma_min": 1e-4,
    }
    decoder_params = {
        "channels": [256, 256],
        "dropout": 0.05,
        "attention_head_dim": 64,
        "n_blocks": 2,
        "num_mid_blocks": 2,
        "num_heads": 2,
        "act_fn": "snakebeta",
    }
    return CFM(
        in_channels=n_feats * 2,
        out_channel=n_feats,
        cfm_params=cfm_params,
        decoder_params=decoder_params,
    ).to(device)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def apply_overrides(h, args):
    if args.input_dir is not None:
        h.test_input_wavs_dir = args.input_dir
    if args.output_dir is not None:
        h.test_wav_output_dir = args.output_dir
    if args.encoder_ckpt is not None:
        h.checkpoint_file_load_Encoder = args.encoder_ckpt
    if args.decoder_ckpt is not None:
        h.checkpoint_file_load_Decoder = args.decoder_ckpt
    if args.cfm_ckpt is not None:
        h.checkpoint_file_load_cfmmodel = args.cfm_ckpt
    if args.n_timesteps is not None:
        h.n_timesteps = args.n_timesteps
    if args.temperature is not None:
        h.temperature = args.temperature
    return h


def inference(h, device):
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    cfmmodel = build_cfm_model(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder["encoder"])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder["decoder"])
    state_dict_cfmmodel = load_checkpoint(h.checkpoint_file_load_cfmmodel, device)
    cfmmodel.load_state_dict(state_dict_cfmmodel["cfmmodel"])

    input_root = Path(h.test_input_wavs_dir)
    wav_output_root = Path(h.test_wav_output_dir)
    filelist = collect_audio_files(input_root)
    if len(filelist) == 0:
        print("No audio files found under '{}'.".format(input_root))
        return

    wav_output_root.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    decoder.eval()
    cfmmodel.eval()

    n_timesteps = int(getattr(h, "n_timesteps", 6))
    temperature = float(getattr(h, "temperature", 1.0))

    total_len = 0.0
    total_time = 0.0

    mdct_operation = MDCT(80).to(device)
    imdct_operation = IMDCT(80).to(device)
    compressor = MDCTAmplitudeCompressor(alpha=0.5, time_freq_dims=(2, 1)).to(device)

    print("Found {} audio files.".format(len(filelist)))
    print("Input root : {}".format(input_root))
    print("Wav output : {}".format(wav_output_root))
    print("Device     : {}".format(device))
    print("CFM steps  : {}".format(n_timesteps))
    print("Temperature: {}".format(temperature))

    with torch.no_grad():
        for i, in_path in enumerate(filelist, 1):
            rel_path = in_path.relative_to(input_root)
            out_wav_path = (wav_output_root / rel_path).with_suffix(".wav")
            out_wav_path.parent.mkdir(parents=True, exist_ok=True)

            raw_wav, _ = librosa.load(str(in_path), sr=h.sampling_rate, mono=True)
            raw_wav = torch.from_numpy(raw_wav).float().to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = perf_counter()

            mdct = mdct_operation(raw_wav.unsqueeze(0)).permute(0, 2, 1)
            latent, _, _ = encoder(mdct)
            _, mdct_g = decoder(latent)
            mask = torch.ones(mdct_g.size(0), 1, mdct_g.size(-1), device=device)
            mdct_g_comp, scale_g = compressor(mdct_g)
            mdct_hat_comp = cfmmodel(mdct_g_comp, mask, n_timesteps, temperature)
            mdct_hat = compressor.invert(mdct_hat_comp, scale_g)
            audio = imdct_operation(mdct_hat.permute(0, 2, 1)).squeeze()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += perf_counter() - t0

            audio_np = audio.detach().cpu().numpy()
            total_len += len(audio_np)
            sf.write(str(out_wav_path), audio_np, h.sampling_rate, "PCM_16")

            print("[{}/{}] {} -> {}".format(i, len(filelist), rel_path, out_wav_path))

    total_audio_seconds = total_len / float(h.sampling_rate)
    generation_time_per_second = total_time / total_audio_seconds if total_audio_seconds > 0 else 0.0

    print("\nDone.")
    print("Total audio duration: {:.3f} s".format(total_audio_seconds))
    print("Total inference time: {:.3f} s".format(total_time))
    print("The time of generating 1s speech is {:.6f} seconds.".format(generation_time_per_second))


def main():
    parser = argparse.ArgumentParser(description="Run CFMDCTCodec inference on a folder of audio files.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--input_dir", default=None, help="Override test_input_wavs_dir.")
    parser.add_argument("--output_dir", default=None, help="Override test_wav_output_dir.")
    parser.add_argument("--encoder_ckpt", default=None, help="Override encoder checkpoint path.")
    parser.add_argument("--decoder_ckpt", default=None, help="Override decoder checkpoint path.")
    parser.add_argument("--cfm_ckpt", default=None, help="Override CFM checkpoint path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--n_timesteps", type=int, default=None, help="Number of CFM Euler steps.")
    parser.add_argument("--temperature", type=float, default=None, help="CFM sampling temperature.")
    args = parser.parse_args()

    print("Initializing Inference Process..")
    with open(args.config, "r", encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    h = apply_overrides(h, args)

    device = resolve_device(args.device)
    torch.manual_seed(h.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(h.seed)

    inference(h, device)


if __name__ == "__main__":
    main()

# CFMDCTCodec 0.65 kbps 16 kHz

CFMDCTCodec is a low-bitrate neural speech codec built around MDCT-domain
analysis, vector quantization, and a conditional flow matching refinement model.
This release contains the 16 kHz, 0.65 kbps configuration.

The default configuration uses:

- Sampling rate: 16 kHz
- MDCT hop size: 40 samples
- Downsampling ratio: 8
- Latent rate: 50 frames/s
- Quantizer: 1 codebook with 8192 entries
- Bitrate: `50 * log2(8192) = 650 bps`

## Repository Layout

```text
.
├── config.json              # Default runnable config with relative paths
├── config.example.json      # Copy/edit this for your own data and checkpoints
├── train.py                 # Training entry point
├── inference.py             # Audio reconstruction entry point
├── models.py
├── quantize.py
├── dataset.py
├── flowmaatching.py
├── fm_decoder.py
└── modules/
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code was developed with PyTorch 2.x. CUDA is recommended for training and
inference, but the inference script also accepts `--device cpu`.

## Data

Training and validation inputs are directories containing audio files. The
default config expects:

```text
data/train_wavs
data/valid_wavs
examples/wavs
```

You can either edit `config.json` or pass overrides to the inference script.
Audio is loaded as mono and resampled to `sampling_rate` from the config.

## Checkpoints

Model checkpoints are not included in this source tree. Place the checkpoint
files under `ckpt/` or pass their paths explicitly.

The default checkpoint names are:

```text
ckpt/encoder_02000000
ckpt/decoder_02000000
ckpt/cfmmodel_02000000
```

Each checkpoint is expected to contain one of the following top-level keys:

```text
encoder
decoder
cfmmodel
```

## Training

Edit `config.json` or create your own config from `config.example.json`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.json
```

Training checkpoints and TensorBoard logs are written to `checkpoint_path`
from the config, which defaults to `ckpt`.

```bash
tensorboard --logdir ckpt/logs
```

## Inference

Run inference with paths from `config.json`:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --config config.json --device cuda
```

Or override paths from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config config.json \
  --input_dir examples/wavs \
  --output_dir outputs \
  --encoder_ckpt ckpt/encoder_02000000 \
  --decoder_ckpt ckpt/decoder_02000000 \
  --cfm_ckpt ckpt/cfmmodel_02000000 \
  --n_timesteps 6 \
  --temperature 1.0 \
  --device cuda
```

CPU inference:

```bash
python inference.py --config config.json --device cpu
```

## Release Checklist

Before publishing this repository, verify:

- The selected open-source license is approved by the code owner.
- Any released checkpoints were trained on data that can be redistributed or
  referenced publicly.
- No private data, paths, logs, or model weights are committed.
- `config.json` contains only relative example paths.
- A small authorized audio example is added if you want an out-of-box demo.

## Acknowledgements

This implementation uses common neural audio codec building blocks and includes
components adapted from or inspired by HiFi-GAN-style discriminators,
ConvNeXt-style temporal blocks, Diffusers transformer blocks, and VQ/VQGAN-style
quantization methods. See `NOTICE.md` for additional notes.

## License

The repository currently includes an MIT license template. Confirm ownership and
third-party compatibility before the public release.

#!/usr/bin/env bash
# Setup Wav2Lip + GFPGAN post-processing (CUDA).
# - Clones Wav2Lip (code only; NEVER pip install its requirements.txt - pins torch 1.1!)
# - Downloads wav2lip_gan.pth (~146MB) + s3fd face detector (~90MB)
# - Installs gfpgan + facexlib (basicsr needs a setup.py patch on py3.13+, handled below)
# Idempotent: safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="$ROOT/third_party/Wav2Lip"
MODELS="$ROOT/models"
cd "$ROOT"

echo "=== [1/4] Wav2Lip repo ==="
if [ ! -f "$THIRD/inference.py" ]; then
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/Rudrabha/Wav2Lip "$THIRD"
else
  echo "exists: $THIRD"
fi

# Compat patches for modern stack (idempotent):
# - librosa>=0.10 made filters.mel() keyword-only
# - torch>=2.6 defaults weights_only=True, but the official GAN checkpoint
#   is a TorchScript archive (trusted: official Drive file, verified size).
python3 - <<'EOF'
from pathlib import Path
audio = Path("third_party/Wav2Lip/audio.py")
src = audio.read_text()
old = "librosa.filters.mel(hp.sample_rate, hp.n_fft,"
if old in src:
    audio.write_text(src.replace(old, "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,"))
    print("patched Wav2Lip audio.py for librosa>=0.10")
inf = Path("third_party/Wav2Lip/inference.py")
src = inf.read_text()
old1 = "checkpoint = torch.load(checkpoint_path)\n"
old2 = "map_location=lambda storage, loc: storage)"
if old1 in src and "weights_only" not in src:
    src = src.replace(old1, "checkpoint = torch.load(checkpoint_path, weights_only=False)\n")
    src = src.replace(old2, "map_location=lambda storage, loc: storage, weights_only=False)")
    inf.write_text(src)
    print("patched Wav2Lip inference.py for torch>=2.6")
else:
    inf = Path("third_party/Wav2Lip/inference.py")
    src2 = inf.read_text()
    marker = "TorchScript, using directly"
    if marker not in src2:
        a = 'model = Wav2Lip()\n\tprint("Load checkpoint from: {}".format(path))\n\tcheckpoint = _load(path)\n\ts = checkpoint["state_dict"]'
        b = ('print("Load checkpoint from: {}".format(path))\n\tcheckpoint = _load(path)\n'
             '\tif isinstance(checkpoint, torch.jit.ScriptModule):\n'
             '\t\tprint("Checkpoint is TorchScript, using directly")\n'
             '\t\treturn checkpoint.to(device).eval()\n\tmodel = Wav2Lip()\n\ts = checkpoint["state_dict"]')
        assert a in src2, "load_model pattern changed"
        inf.write_text(src2.replace(a, b))
        print("patched load_model for scripted GAN checkpoint")
    print("inference.py already patched")
EOF

echo "=== [2/4] Python deps (gfpgan stack + gdown, torch untouched) ==="
# basicsr 1.4.2 sdist has a broken setup.py on py3.13 (exec/locals KeyError):
# patch the GitHub checkout and install with --no-build-isolation.
if ! python3 -c "import basicsr" 2>/dev/null; then
  rm -rf /tmp/basicsr && git clone --depth 1 https://github.com/XPixelGroup/BasicSR /tmp/basicsr
  python3 - <<'EOF'
from pathlib import Path
p = Path("/tmp/basicsr/setup.py")
src = p.read_text()
p.write_text(src.replace("return locals()['__version__']", "return '1.4.2'"))
print("patched basicsr setup.py")
EOF
  pip install /tmp/basicsr --no-build-isolation
fi
pip install gfpgan --no-deps facexlib gdown
python3 -c "from gfpgan import GFPGANer; print('GFPGANer import OK')"

echo "=== [3/4] Checkpoints ==="
mkdir -p "$MODELS" "$THIRD/face_detection/detection/sfd"
if [ ! -f "$MODELS/wav2lip_gan.pth" ]; then
  gdown -O "$MODELS/wav2lip_gan.pth" \
    "https://drive.google.com/uc?id=15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"
else
  echo "exists: $MODELS/wav2lip_gan.pth"
fi
if [ ! -f "$THIRD/face_detection/detection/sfd/s3fd.pth" ]; then
  curl -sL -o "$THIRD/face_detection/detection/sfd/s3fd.pth" \
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
else
  echo "exists: s3fd.pth"
fi
ls -la "$MODELS/wav2lip_gan.pth" "$THIRD/face_detection/detection/sfd/s3fd.pth"

echo "=== [4/4] GFPGAN weights (auto-download on first run) ==="
echo "GFPGANv1.4.pth will be fetched to $MODELS on first lip-sync job."
echo "DONE. Enable per-job via /dub [khop_moi] or global WAV2LIP_ENABLED in .env"

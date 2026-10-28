#!/usr/bin/env bash
# Install into a dedicated directory; does not start training or touch hardware.
set -euo pipefail
SO101_ROOT="${SO101_ROOT:-/opt/so101}"
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MAMBA_ROOT_PREFIX="$SO101_ROOT/mamba"
export PIP_CACHE_DIR="$SO101_ROOT/cache/pip"
export HF_HOME="$SO101_ROOT/cache/huggingface"
mkdir -p "$SO101_ROOT/bin" "$SO101_ROOT/logs" "$PIP_CACHE_DIR"
if [[ ! -x "$SO101_ROOT/bin/micromamba" ]]; then
    # Some VM networks cannot reach api.anaconda.org after the redirect.
    if curl -fsSL --connect-timeout 15 --max-time 120 \
        https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$SO101_ROOT/micromamba.tar.bz2"; then
        tar -xjf "$SO101_ROOT/micromamba.tar.bz2" -C "$SO101_ROOT" bin/micromamba
    else
        curl -fsSL --retry 2 --connect-timeout 15 --max-time 120 \
            https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-linux-64 \
            -o "$SO101_ROOT/bin/micromamba"
        chmod +x "$SO101_ROOT/bin/micromamba"
    fi
fi
if [[ ! -x "$SO101_ROOT/env/bin/python" ]]; then
    "$SO101_ROOT/bin/micromamba" create -y -p "$SO101_ROOT/env" -c conda-forge python=3.10 pip 'ffmpeg=7.1.*'
fi
if [[ ! -d "$SO101_ROOT/so101-pi05-base/.git" ]]; then
    git clone https://github.com/jinnymo/so101-pi05-base.git "$SO101_ROOT/so101-pi05-base"
    git -C "$SO101_ROOT/so101-pi05-base" checkout 73167f4dba88633aeec95ba7a3b3881916fa9aaa
fi
if [[ ! -d "$SO101_ROOT/vlash/.git" ]]; then
    git clone https://github.com/mit-han-lab/vlash.git "$SO101_ROOT/vlash"
    git -C "$SO101_ROOT/vlash" checkout 22cbabfee0f57874987c75a35a7dac129e695db0
fi
[[ "$(git -C "$SO101_ROOT/so101-pi05-base" rev-parse HEAD)" == 73167f4dba88633aeec95ba7a3b3881916fa9aaa ]] || {
    printf '%s\n' 'Unexpected author repository revision; use a separate SO101_ROOT.' >&2; exit 1;
}
[[ "$(git -C "$SO101_ROOT/vlash" rev-parse HEAD)" == 22cbabfee0f57874987c75a35a7dac129e695db0 ]] || {
    printf '%s\n' 'Unexpected VLASH revision; use a separate SO101_ROOT.' >&2; exit 1;
}
"$SO101_ROOT/env/bin/python" "$PACKAGE_DIR/apply_runtime.py" --root "$SO101_ROOT"
"$SO101_ROOT/env/bin/python" -m pip install -c "$PACKAGE_DIR/constraints.txt" \
    -e "$SO101_ROOT/vlash" torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
    grpcio==1.73.1 grpcio-tools==1.73.1 grpcio-reflection==1.73.1 \
    huggingface-hub==0.35.3 pyarrow==24.0.0
"$SO101_ROOT/env/bin/python" - "$SO101_ROOT" <<'PY'
import importlib.util
import shutil
import sys
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.find_spec('lerobot')
target = Path(spec.origin).parent / 'datasets/video_utils.py'
shutil.copy2(root / 'so101-pi05-base/training-docker/patched/video_utils.py', target)
print('Patched LeRobot video decoding:', target)
PY
"$SO101_ROOT/env/bin/python" -m pip freeze > "$SO101_ROOT/logs/installed-requirements.txt"
printf '%s\n' 'Environment installed. Next: python server/download_assets.py --root /opt/so101'

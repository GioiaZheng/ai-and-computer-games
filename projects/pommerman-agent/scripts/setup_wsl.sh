#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-pommerman}"
PLAYGROUND_COMMIT="5315f6da378f495737dfe34a4ba7f50c84423ce7"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.8 pip -y
fi
conda activate "$ENV_NAME"

python -m pip install -r "$PROJECT_DIR/requirements.txt"
python -m pip install \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

mkdir -p "$PROJECT_DIR/.vendor"
if [[ ! -d "$PROJECT_DIR/.vendor/playground/.git" ]]; then
    git clone https://github.com/MultiAgentLearning/playground.git \
        "$PROJECT_DIR/.vendor/playground"
fi
git -C "$PROJECT_DIR/.vendor/playground" checkout "$PLAYGROUND_COMMIT"
python -m pip install -e "$PROJECT_DIR/.vendor/playground" --no-deps

python -c "import pommerman, torch; print('Pommerman:', pommerman.REGISTRY[0]); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

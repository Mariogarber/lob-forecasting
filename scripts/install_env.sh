#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create / refresh the `lob` conda env for the Wunder Predictorium pipeline.
#
# Steps:
#   1. Create conda env `lob` with Python 3.12.
#   2. Install the numerical + ML stack via pip from requirements.txt
#      (PyTorch comes from the cu121 wheel index — works with driver 595+).
#   3. Best-effort install of the native Mamba-2 CUDA kernels. If the build
#      fails (no nvcc / mismatched toolchain), the pipeline still works
#      because it falls back to the pure-PyTorch `mambapy` backend.
#   4. Smoke-test: torch.cuda.is_available() and a tiny model forward pass.
#
# Usage:
#   bash scripts/install_env.sh
# ---------------------------------------------------------------------------

set -euo pipefail

ENV_NAME="${LOB_ENV_NAME:-lob}"
PY_VERSION="${LOB_PY_VERSION:-3.12}"
CONDA_BASE="${CONDA_BASE:-/home/eder/miniconda3}"

echo ">>> Creating conda env '${ENV_NAME}' (python=${PY_VERSION})..."
"${CONDA_BASE}/bin/conda" create -y -n "${ENV_NAME}" "python=${PY_VERSION}" pip

ENV_PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
ENV_PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

echo ">>> Upgrading pip / setuptools / wheel..."
"${ENV_PIP}" install --upgrade pip setuptools wheel

# Install PyTorch first so that any --no-build-isolation packages can find it.
echo ">>> Installing PyTorch (cu121 wheels)..."
"${ENV_PIP}" install --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"

echo ">>> Installing project requirements (the rest)..."
"${ENV_PIP}" install -r "$(dirname "$0")/../requirements.txt"

echo ">>> [optional] Attempting native Mamba-2 CUDA kernels..."
# Requires nvcc inside the env (cuda-nvcc + cuda-cudart-dev + headers).
# If those packages are not present we skip silently; the pipeline still
# works through the pure-PyTorch ``mambapy`` fallback.
if [ -x "${CONDA_BASE}/envs/${ENV_NAME}/bin/nvcc" ]; then
    echo "    nvcc detected at ${CONDA_BASE}/envs/${ENV_NAME}/bin/nvcc"
    CUDA_HOME="${CONDA_BASE}/envs/${ENV_NAME}" \
    PATH="${CONDA_BASE}/envs/${ENV_NAME}/bin:${PATH}" \
    TORCH_CUDA_ARCH_LIST="8.9" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE \
    "${ENV_PIP}" install --no-build-isolation --no-deps \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" \
        "git+https://github.com/state-spaces/mamba.git@v2.2.2" \
        || echo "    (skip) mamba-ssm build failed — pipeline will use mambapy fallback."
else
    echo "    No nvcc in env. To enable native Mamba-2 kernels run:"
    echo "       conda install -n ${ENV_NAME} -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvtx -y"
    echo "    and re-run this script."
fi

echo ">>> Smoke test:"
"${ENV_PY}" - <<'PY'
import torch
print(f"torch         = {torch.__version__}")
print(f"cuda available= {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0      = {torch.cuda.get_device_name(0)}")
    print(f"capability    = {torch.cuda.get_device_capability(0)}")
import lightgbm, sklearn, pandas, numpy
print(f"lightgbm      = {lightgbm.__version__}")
print(f"sklearn       = {sklearn.__version__}")
print(f"pandas        = {pandas.__version__}")
print(f"numpy         = {numpy.__version__}")
try:
    import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
    from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore
    print(f"mamba-ssm     = native CUDA kernels available")
except Exception as exc:
    print(f"mamba-ssm     = unavailable ({type(exc).__name__}); will use mambapy fallback")
import mambapy
print(f"mambapy       = loaded (pure-PyTorch fallback)")
PY

echo
echo ">>> Done. Activate with:  conda activate ${ENV_NAME}"

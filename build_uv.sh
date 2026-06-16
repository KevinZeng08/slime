#!/bin/bash

# Build slime from source into an existing uv-managed virtualenv.
#
# Target setting (this machine):
#   - CUDA 13.1 (nvcc release 13.1, CUDA_HOME=/usr/local/cuda-13.1)
#   - NVIDIA H800 (Hopper / SM90)
#   - sglang v0.5.13 (slime upstream just bumped to it)
#   - uv-managed venv at rl/slime/.venv (Python 3.12)
#
# This mirrors build_conda.sh, but:
#   - uses `uv pip` against an existing venv instead of micromamba;
#   - follows the Dockerfile's ENABLE_CUDA_13=1 path (cu130 wheels, TE from
#     source, cu130 sgl-kernel, fzyzcjy triton fork).
#
# Versions are kept in sync with docker/Dockerfile and docker/version.txt.

set -ex

# ======================================== Config =============================================

# Resolve slime dir from this script's location so it works regardless of CWD.
SLIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Sibling clones (sglang / Megatron-LM) go next to slime by default.
BASE_DIR="${BASE_DIR:-$(dirname "$SLIME_DIR")}"

# The uv venv to install into. Override VENV_DIR to point elsewhere.
VENV_DIR="${VENV_DIR:-$SLIME_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"

# Keep these in sync with docker/Dockerfile:
SGLANG_VERSION="${SGLANG_VERSION:-v0.5.13}"
MEGATRON_COMMIT="${MEGATRON_COMMIT:-1dcf0dafa884ad52ffb243625717a3471643e087}"
PATCH_VERSION="${PATCH_VERSION:-latest}"

# CUDA 13 toolchain knobs.
TORCH_CUDA="${TORCH_CUDA:-cu130}"                 # pytorch wheel channel
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
# sgl-kernel is NOT pinned here: sglang's `python[all]` install pulls the matching
# sglang-kernel==0.4.3 (cu130, torch-2.11 ABI). See the sglang install section.
SGL_WHL_INDEX="https://docs.sglang.ai/whl/${TORCH_CUDA}/"

# Optional FlashAttention-3 (Hopper) build. Slow (~tens of minutes). H800 is
# SM90 so it is useful, but off by default to keep the build short.
INSTALL_FA3="${INSTALL_FA3:-0}"
FA3_COMMIT="fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89"

# sglang patch: the cu13 image build sets ENABLE_SGLANG_PATCH=0; we default to 1
# but guard the apply, so a non-applicable patch is skipped instead of failing.
ENABLE_SGLANG_PATCH="${ENABLE_SGLANG_PATCH:-1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-64}"

# ======================================== Helpers =============================================

# All installs target the venv explicitly so we never need to `activate` it.
uvpip() { uv pip install --python "$PYTHON" "$@"; }

# ======================================== Preflight =============================================

command -v uv >/dev/null 2>&1 || { echo "uv not found in PATH" >&2; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found; need CUDA toolkit" >&2; exit 1; }
[ -x "$PYTHON" ] || { echo "venv python not found at $PYTHON (create it with: uv venv \"$VENV_DIR\" --python 3.12)" >&2; exit 1; }

# sglang's editable install builds a Rust extension (sglang-grpc via
# setuptools-rust), so we need a working rustc + cargo.
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

# sglang-grpc's build.rs compiles .proto via tonic-build/prost-build, which
# requires the `protoc` binary.
sudo apt-get update && sudo apt-get install -y protobuf-compiler
command -v protoc >/dev/null 2>&1 || { echo "protoc not found after install attempt" >&2; exit 1; }

# Build-time deps that --no-build-isolation builds expect to find in the env.
uvpip cuda-python==13.0 setuptools wheel "packaging>=24.2" pybind11 cmake ninja

# ======================================== PyTorch (cu130) =============================================

# Pin torch on the cu130 channel up front so every --no-build-isolation build
# below compiles against the right ABI. Adjust the pin if 2.11.0 is unavailable
# torchvision MUST be pinned to the matching release
# (torch 2.11 <-> torchvision 0.26; leaving it unpinned pulls 0.27 which is built
# for torch 2.12 and fails at import with "operator torchvision::nms does not exist").
uvpip --force-reinstall \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url "$TORCH_INDEX"

# ======================================== sglang (from source) =============================================

if [ ! -d "$BASE_DIR/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git "$BASE_DIR/sglang"
fi
cd "$BASE_DIR/sglang"
git fetch --tags --quiet || true
git checkout "${SGLANG_VERSION}"

# Install sglang (python[all]) but keep torch from being downgraded off cu130.
uvpip -e "python[all]" --extra-index-url "$TORCH_INDEX"

# Re-pin torch (sglang may have pulled a default-channel build).
#
# NOTE: do NOT force-reinstall the old sgl_kernel 0.3.17.post2 wheel here. The
# `python[all]` install above already pulls sglang's pinned sglang-kernel==0.4.3,
# whose cu130 wheel is built against torch 2.11's ABI. The 0.3.17.post2 wheel is
# compiled against an older torch (c10::cuda::c10_cuda_check_implementation takes
# `int`) and is ABI-incompatible with torch 2.11.0 (which takes `unsigned int`);
# because both packages write to the same sgl_kernel/ dir, force-installing it
# overwrites the good .so and breaks `import sgl_kernel` with an undefined-symbol
# error. The Dockerfile only needs it because it starts from a cu129 base image
# with an older torch; this from-scratch build pins torch 2.11.0 instead.
uvpip --force-reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url "$TORCH_INDEX"
# deep-gemm is optional; best-effort on the cu130 channel.
uvpip --force-reinstall --no-deps sgl-deep-gemm --index-url "$SGL_WHL_INDEX" || true

# ======================================== Native python deps =============================================

# flash-attn 2 (TE caps the supported FA2 version at 2.7.4.post1).
MAX_JOBS="$MAX_JOBS" uvpip flash-attn==2.7.4.post1 --no-build-isolation

# flash-attn 3 (Hopper) - optional, slow.
if [ "$INSTALL_FA3" = "1" ]; then
  cd "$BASE_DIR"
  rm -rf flash-attention
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout "$FA3_COMMIT"
  git submodule update --init
  cd hopper
  MAX_JOBS=96 "$PYTHON" setup.py install
  py_site="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
  mkdir -p "$py_site/flash_attn_3"
  cp flash_attn_interface.py "$py_site/flash_attn_3/flash_attn_interface.py"
  cd "$BASE_DIR" && rm -rf flash-attention
fi

uvpip git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
uvpip flash-linear-attention==0.4.1
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
uvpip git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
uvpip tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# TransformerEngine: no cu13 wheel yet, build from source (matches Dockerfile).
# nvidia-mathdx is left unpinned (as in TE CI): 26.6.0 does not exist on PyPI /
# pypi.nvidia.com (per-arch max is 25.6.0), and TE release_v2.10 does not pin or
# reference it (links plain CUDA::cublas), so picking the latest is safe.
uvpip nvidia-mathdx
# TE's common/util/logging.h does `#include "nccl.h"` but its CMake never adds an
# NCCL include dir: upstream relies on the CUDA base image's system libnccl-dev
# (/usr/include/nccl.h). On a bare host that header is missing, so point the
# compiler at the nvidia-nccl wheel's headers via CPATH (header-only; TE does not
# link NCCL). nvidia-nccl-cu13 is already pulled in by torch (cu130).
NCCL_INC="$("$PYTHON" -c 'import os,nvidia.nccl as n; print(os.path.join(list(n.__path__)[0], "include"))')"
[ -f "$NCCL_INC/nccl.h" ] || { echo "nccl.h not found under $NCCL_INC" >&2; exit 1; }
CPATH="$NCCL_INC${CPATH:+:$CPATH}" \
  uvpip --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10

# apex (fused optimizers / layernorm). Built from a local clone so we can relax
# apex's CUDA guard: it aborts on ANY nvcc-vs-torch CUDA difference, but here nvcc
# is 13.1 while torch is cu130 (13.0) and only a 13.1 toolkit is installed. A 13.x
# minor mismatch is benign (apex's own error says so), so downgrade the guard to
# compare major versions only.
# NOTE: uv's CLI (clap) treats an option value starting with "--" as a new flag,
# so --config-settings must attach its value with "=" (unlike pip). setuptools
# shlex-splits it, so "--cpp_ext --cuda_ext --parallel 8" becomes separate options.
APEX_COMMIT="10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
cd "$BASE_DIR"
rm -rf apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout "$APEX_COMMIT"
sed -i 's/if (bare_metal_version != torch_binary_version):/if (bare_metal_version.major != torch_binary_version.major):/' setup.py
NVCC_APPEND_FLAGS="--threads 4" \
  uvpip --no-build-isolation \
  --config-settings="--build-option=--cpp_ext --cuda_ext --parallel 8" \
  ./
cd "$BASE_DIR" && rm -rf apex

# torch_memory_saver (build the cu13 preload .so).
TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$("$PYTHON" -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
uvpip git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation

uvpip git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
uvpip "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# Triton: cu13 fix from masahi, not yet in a Triton release (Dockerfile cu13 path).
if [ ! -d "$BASE_DIR/triton" ]; then
  git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git "$BASE_DIR/triton"
fi
cd "$BASE_DIR/triton"
uvpip -r python/requirements.txt
uvpip --verbose -e . --no-build-isolation

# ======================================== Megatron-LM (from source) =============================================

if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive "$BASE_DIR/Megatron-LM"
fi
cd "$BASE_DIR/Megatron-LM"
git checkout "${MEGATRON_COMMIT}"
# --no-build-isolation so setup.py's helpers_cpp C++ ext builds against the
# current env's pybind11 (otherwise it is silently skipped).
uvpip -e . --no-build-isolation

# ======================================== slime =============================================

cd "$SLIME_DIR"
# Pure-python runtime deps first, then slime itself with --no-deps so the pinned
# cu130 native libs (torch / sgl-kernel / ...) are not re-resolved away.
uvpip -r requirements.txt
uvpip -e . --no-deps

# int4_qat kernel
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
uvpip . --no-build-isolation

# numpy 1.x for megatron; kernels<0.15 so `import sglang` works at runtime.
uvpip "numpy<2"
uvpip "kernels<0.15.0"

# slime-flavored sglang router.
uvpip --force-reinstall \
  https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-1117d05/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
"$PYTHON" -c "import sglang_router; assert 'slime' in sglang_router.__version__"

# ======================================== Patches =============================================

cd "$BASE_DIR/sglang"
if [ "$ENABLE_SGLANG_PATCH" = "1" ] && [ -f "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" ]; then
  if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" 2>/dev/null; then
    git update-index --refresh || true
    git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/sglang.patch" --3way
    if grep -R -n '^<<<<<<< ' .; then
      echo "sglang patch failed to apply cleanly. Please resolve conflicts." >&2
      exit 1
    fi
  else
    echo "sglang patch already applied or not applicable, skipping"
  fi
fi

cd "$BASE_DIR/Megatron-LM"
if [ -f "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" ]; then
  if git apply --check "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" 2>/dev/null; then
    git update-index --refresh || true
    git apply "$SLIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch" --3way
    if grep -R -n '^<<<<<<< ' .; then
      echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
      exit 1
    fi
  else
    echo "megatron patch already applied or not applicable, skipping"
  fi
fi

echo "slime build (uv / CUDA ${TORCH_CUDA} / sglang ${SGLANG_VERSION}) complete -> $VENV_DIR"

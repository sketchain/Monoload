#!/usr/bin/env bash
# Run a command inside the locked ComfyUI image with Monoload mounted as a custom node.
#   MODELS=/path/to/models tests/docker_run.sh python tests/test_lora_hot.py
# $MODELS must contain checkpoints/, diffusion_models/, text_encoders/, loras/ (vae/ is mounted when present).
set -euo pipefail
IMG="docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:?set MODELS to the host models directory}"
mounts=()
for d in checkpoints diffusion_models text_encoders loras vae; do
  [ -d "$MODELS/$d" ] && mounts+=(-v "$MODELS/$d":/opt/ComfyUI/models/$d:ro)
done
exec docker run --rm --network none ${DOCKER_EXTRA:-} \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MONOLOAD="${MONOLOAD:-}" -e MONOLOAD_LANG="${MONOLOAD_LANG:-}" -e MONOLOAD_DISABLE="${MONOLOAD_DISABLE:-}" -e MONOLOAD_KEEP_LORA="${MONOLOAD_KEEP_LORA:-}" -e MONOLOAD_EXACT="${MONOLOAD_EXACT:-}" \
  -e TEST_BROKEN_VAE_API="${TEST_BROKEN_VAE_API:-}" -e MONOLOAD_DISABLE_VAE="${MONOLOAD_DISABLE_VAE:-}" -e MONOLOAD_VAE_WORKSPACE="${MONOLOAD_VAE_WORKSPACE:-}" \
  -e MONOLOAD_DISABLE_VAE_STRIPE="${MONOLOAD_DISABLE_VAE_STRIPE:-}" -e MONOLOAD_VAE_BUDGET="${MONOLOAD_VAE_BUDGET:-}" -e MONOLOAD_VAE_STRIPE_ROWS="${MONOLOAD_VAE_STRIPE_ROWS:-}" -e MONOLOAD_VAE_GN_SCHEME="${MONOLOAD_VAE_GN_SCHEME:-}" \
  -v "$REPO":/opt/ComfyUI/custom_nodes/monoload:ro "${mounts[@]}" \
  -w /opt/ComfyUI/custom_nodes/monoload \
  "$IMG" "$@"

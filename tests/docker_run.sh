#!/usr/bin/env bash
# Run a command inside the locked ComfyUI image with Monoload mounted as a custom node.
#   MODELS=/path/to/models tests/docker_run.sh python -m monoload.convert ... -- --cpu --fp16-unet
# $MODELS must contain diffusion_models/, loras/, monoload/.
set -euo pipefail
IMG="docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${MODELS:?set MODELS to the host models directory}"
exec docker run --rm --network none ${DOCKER_EXTRA:-} \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e MONOLOAD_STAGING="${MONOLOAD_STAGING:-auto}" -e MONOLOAD_BUFFER_MB="${MONOLOAD_BUFFER_MB:-}" \
  -v "$REPO":/opt/ComfyUI/custom_nodes/monoload:ro \
  -v "$MODELS/diffusion_models":/opt/ComfyUI/models/diffusion_models:ro \
  -v "$MODELS/loras":/opt/ComfyUI/models/loras:ro \
  -v "$MODELS/monoload":/opt/ComfyUI/models/monoload \
  -w /opt/ComfyUI/custom_nodes/monoload \
  "$IMG" "$@"

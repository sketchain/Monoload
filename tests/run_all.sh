#!/usr/bin/env bash
# Full CPU test suite in the locked image. Needs, under $MODELS:
#   checkpoints/v1-5-pruned-emaonly-fp16.safetensors        (CheckpointLoaderSimple)
#   diffusion_models/v1-5-pruned-emaonly-fp16.safetensors   (UNETLoader; same file, a hard link is fine)
#   text_encoders/clip_l.safetensors                        (CLIPLoader)
#   loras/{rubber_duck,lycoris_annalise,synthetic_lokr_sd15,synthetic_loha_sd15}.safetensors
# (see README "测试" for download links; the synthetic ones come from tests/make_synthetic_loras.py)
set -uo pipefail
cd "$(dirname "$0")/.."
export MODELS="${MODELS:?set MODELS}"
R=tests/docker_run.sh
SD=v1-5-pruned-emaonly-fp16.safetensors
fails=0
step() { echo; echo "######## $*"; }
run() { "$@" 2>&1 | grep -vE "agent.cpp|sysfs nodes|comfy_kitchen backend|it/s\]|s/it\]" ; local rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { echo "!!! exit $rc"; fails=$((fails+1)); }; }

step "plugin entry (installed)";          run $R python tests/test_entry.py
step "plugin entry (MONOLOAD_DISABLE=1)"; run env MONOLOAD_DISABLE=1 $R python tests/test_entry.py
step "dtype paths vs native numerics";    run $R python tests/test_dtype_paths.py
step "LoRA: CheckpointLoaderSimple, UNETLoader + CLIPLoader, hooks, refusals"
run $R env COMFY_ARGS="--cpu --fp16-unet" python tests/test_lora_hot.py --checkpoint $SD --unet $SD --clip clip_l.safetensors
echo; echo "######## suites failed: $fails"
exit $fails

#!/usr/bin/env bash
# Full CPU test suite in the locked image. Needs, under $MODELS:
#   diffusion_models/v1-5-pruned-emaonly-fp16.safetensors
#   loras/{rubber_duck,lycoris_annalise,synthetic_lokr_sd15,synthetic_loha_sd15}.safetensors
# (see README "测试" for download links; the synthetic ones come from tests/make_synthetic_loras.py)
set -uo pipefail
cd "$(dirname "$0")/.."
export MODELS="${MODELS:?set MODELS}"
R=tests/docker_run.sh
SRC=v1-5-pruned-emaonly-fp16.safetensors
CONV=sd15-fp16.safetensors
ARGS="--cpu --disable-mmap --fp16-unet"
LORAS=rubber_duck.safetensors,lycoris_annalise.safetensors,synthetic_lokr_sd15.safetensors,synthetic_loha_sd15.safetensors
fails=0
step() { echo; echo "######## $*"; }
run() { "$@" 2>&1 | grep -vE "agent.cpp|sysfs nodes|comfy_kitchen backend|it/s\]|s/it\]" ; local rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { echo "!!! exit $rc"; fails=$((fails+1)); }; }

step "transfer unit test";              run $R python tests/test_transfer_unit.py
step "convert ($ARGS)";                 run $R python -m monoload.convert /opt/ComfyUI/models/diffusion_models/$SRC -o /opt/ComfyUI/models/monoload/$CONV --force -- $ARGS
step "equivalence";                     run $R env COMFY_ARGS="$ARGS" python tests/test_equivalence.py --family sd15 --source $SRC --converted $CONV
step "equivalence, staged 4 MiB";       run env MONOLOAD_STAGING=always MONOLOAD_BUFFER_MB=4 $R env COMFY_ARGS="$ARGS" python tests/test_equivalence.py --family sd15 --source $SRC --converted $CONV --no-sample
step "memory: native";                  run $R env COMFY_ARGS="$ARGS" python tests/test_memory.py --mode native --name $SRC --lora lycoris_annalise.safetensors
step "memory: monoload direct";         run $R env COMFY_ARGS="$ARGS" python tests/test_memory.py --mode monoload --name $CONV --lora lycoris_annalise.safetensors
step "memory: monoload staged";         run env MONOLOAD_STAGING=always $R env COMFY_ARGS="$ARGS" python tests/test_memory.py --mode monoload --name $CONV
step "validation";                      run $R env COMFY_ARGS="$ARGS" python tests/test_validation.py --converted $CONV
step "LoRA";                            run $R env COMFY_ARGS="$ARGS" python tests/test_lora.py --source $SRC --converted $CONV --loras $LORAS --hook-lora rubber_duck.safetensors
echo; echo "######## suites failed: $fails"
exit $fails

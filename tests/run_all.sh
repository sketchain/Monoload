#!/usr/bin/env bash
# Full CPU test suite in the locked image. Needs, under $MODELS:
#   checkpoints/v1-5-pruned-emaonly-fp16.safetensors        (CheckpointLoaderSimple)
#   diffusion_models/v1-5-pruned-emaonly-fp16.safetensors   (UNETLoader; same file, a hard link is fine)
#   diffusion_models/sd15_unet_fp8_scaled.safetensors       (tests/make_fp8_unet.py)
#   text_encoders/clip_l.safetensors                        (CLIPLoader)
#   loras/{rubber_duck,lycoris_annalise,synthetic_lokr_sd15,synthetic_loha_sd15,synthetic_unet_only_sd15}.safetensors
# (see README "测试" for download links; the synthetic ones come from tests/make_synthetic_loras.py)
set -uo pipefail
cd "$(dirname "$0")/.."
export MODELS="${MODELS:?set MODELS}"
OUT="${OUT:-$(mktemp -d)}"; chmod 777 "$OUT"
R=tests/docker_run.sh
SD=v1-5-pruned-emaonly-fp16.safetensors
ARGS="--cpu --fp16-unet"
fails=0
step() { echo; echo "######## $*"; }
run() { "$@" 2>&1 | grep -vE "agent.cpp|sysfs nodes|comfy_kitchen backend|it/s\]|s/it\]|nodes_replacements" ; local rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { echo "!!! exit $rc"; fails=$((fails+1)); }; }

step "plugin entry (installed)";            run $R python tests/test_entry.py
step "plugin entry (MONOLOAD_DISABLE=1)";   run env MONOLOAD_DISABLE=1 $R python tests/test_entry.py
step "plugin entry (MONOLOAD_KEEP_LORA=1)"; run env MONOLOAD_KEEP_LORA=1 $R python tests/test_entry.py
step "plugin entry (MONOLOAD_EXACT=1)";     run env MONOLOAD_EXACT=1 $R python tests/test_entry.py
# every functional suite runs on both merge paths: bit-exact (MONOLOAD_EXACT=1)
# and the default (fused / relaxed, checked against native within tolerance)
for EXACT in 1 ""; do
  M=$([ -n "$EXACT" ] && echo "MONOLOAD_EXACT=1" || echo "default path")
  E="env MONOLOAD_EXACT=$EXACT"
  step "[$M] dtype paths (incl. fp8)"; run $E $R python tests/test_dtype_paths.py
  step "[$M] LoRA: CheckpointLoaderSimple, UNETLoader + CLIPLoader, hooks, merge, refusals"
  run $E $R env COMFY_ARGS="$ARGS" python tests/test_lora_hot.py --checkpoint $SD --unet $SD --clip clip_l.safetensors
  step "[$M] fp8 model + LoRA (relaxed merge)"
  run $E $R env COMFY_ARGS="$ARGS" python tests/test_quant.py --unet sd15_unet_fp8_scaled.safetensors --clip clip_l.safetensors
  REF=/out/ref${EXACT:+_exact}.pt
  step "[$M] per-prompt release: reference from a process that never saw a LoRA"
  run $E DOCKER_EXTRA="-v $OUT:/out" $R env COMFY_ARGS="$ARGS" python tests/test_release.py --save-reference $REF
  for c in ram_pressure classic lru; do
    step "[$M] per-prompt release ($c cache)"
    run $E DOCKER_EXTRA="-v $OUT:/out" $R env COMFY_ARGS="$ARGS" python tests/test_release.py --reference $REF --cache $c
  done
  step "[$M] MONOLOAD_KEEP_LORA=1"
  run $E MONOLOAD_KEEP_LORA=1 DOCKER_EXTRA="-v $OUT:/out" $R env COMFY_ARGS="$ARGS" python tests/test_release.py --reference $REF
done
echo; echo "######## suites failed: $fails"
exit $fails

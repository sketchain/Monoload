#!/bin/sh
# Sample GTT usage and this cgroup's memory every $INTERVAL seconds (default 0.5).
# Prints current values plus the running maximum. Run it inside the ComfyUI
# container (docker exec) or on the PVE host (set CGROUP=/sys/fs/cgroup/lxc/700).
#   docker exec comfyui sh /opt/ComfyUI/custom_nodes/monoload/tools/watch_mem.sh
INTERVAL="${INTERVAL:-0.5}"
CGROUP="${CGROUP:-/sys/fs/cgroup}"
gtt() {
  t=0
  for f in /sys/class/drm/card[0-9]*/device/mem_info_gtt_used; do
    case "$f" in *-*) continue;; esac
    [ -r "$f" ] && t=$((t + $(cat "$f")))
  done
  echo "$t"
}
g0=$(gtt); c0=$(cat "$CGROUP/memory.current" 2>/dev/null || echo 0)
gmax=$g0; cmax=$c0
echo "baseline: GTT $(awk "BEGIN{printf \"%.2f\", $g0/2^30}") GiB, cgroup $(awk "BEGIN{printf \"%.2f\", $c0/2^30}") GiB  (Ctrl-C to stop)"
while true; do
  g=$(gtt); c=$(cat "$CGROUP/memory.current" 2>/dev/null || echo 0)
  [ "$g" -gt "$gmax" ] && gmax=$g
  [ "$c" -gt "$cmax" ] && cmax=$c
  awk -v g="$g" -v gm="$gmax" -v g0="$g0" -v c="$c" -v cm="$cmax" -v c0="$c0" -v t="$(date +%T)" 'BEGIN{
    printf "%s GTT %6.2f GiB (peak +%6.2f)   cgroup %6.2f GiB (peak +%6.2f)\n", t, g/2^30, (gm-g0)/2^30, c/2^30, (cm-c0)/2^30 }'
  sleep "$INTERVAL"
done

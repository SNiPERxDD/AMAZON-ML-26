#!/usr/bin/env bash
# Run a command at low priority with a resident-memory ceiling.
#
# Usage: tools/run_guarded.sh [--max-gb N] [--max-swap-gb N] [--log FILE] -- <command...>
# The command's process tree is polled every 2 seconds. It receives SIGTERM when
# its combined RSS exceeds the ceiling, free system memory drops below 1.5 GB, or
# system swap grows more than --max-swap-gb above its level at start (RSS misses
# compressed, swapped and GPU memory on macOS),
# and SIGKILL 10 seconds later if it is still alive. Output goes to the log file;
# the last 40 lines are printed on exit together with the peak RSS.
set -u
max_gb=12
max_swap_gb=4
log_file="$(mktemp -t guarded.XXXXXX)"
while [ $# -gt 0 ]; do
    case "$1" in
        --max-gb) max_gb="$2"; shift 2 ;;
        --log) log_file="$2"; shift 2 ;;
        --max-swap-gb) max_swap_gb="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-4}"
max_kb=$((max_gb * 1024 * 1024))
page_kb=$(( $(pagesize) / 1024 ))

tree_pids() {
    # The root process and all descendants.
    local pids="$1" all="$1" children
    while [ -n "$pids" ]; do
        children=$(pgrep -P "$(echo "$pids" | tr ' ' ',')" 2>/dev/null | tr '\n' ' ')
        all="$all $children"; pids="$children"
    done
    echo $all
}

tree_rss_kb() {
    # Sum RSS of the root process and all descendants.
    ps -o rss= -p "$(tree_pids "$1" | tr ' ' ',')" 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

os_name=$(uname -s)
total_memory_kb=0
if [ "$os_name" = Darwin ]; then
    total_memory_kb=$(( $(sysctl -n hw.memsize) / 1024 ))
fi

free_kb() {
    if [ "$os_name" = Darwin ]; then
        memory_pressure | awk -v total="$total_memory_kb" \
            '/System-wide memory free percentage:/ {gsub(/%/, "", $NF); printf "%.0f\n", total*$NF/100}'
    else
        vm_stat | awk -v p="$page_kb" '/Pages free|Pages inactive|Pages speculative/ {gsub(/\./,"",$NF); s+=$NF} END {print s*p}'
    fi
}

swap_mb() {
    if [ "$os_name" = Darwin ]; then
        sysctl -n vm.swapusage | awk '{gsub(/M/, "", $6); printf "%.0f\n", $6}'
    else
        free -m | awk '/Swap/ {print $3}'
    fi
}
swap_start=$(swap_mb)

nice -n 10 "$@" >"$log_file" 2>&1 &
pid=$!
peak=0; reason=""
while kill -0 "$pid" 2>/dev/null; do
    rss=$(tree_rss_kb "$pid")
    [ "$rss" -gt "$peak" ] && peak=$rss
    if [ "$rss" -gt "$max_kb" ]; then reason="RSS above ${max_gb} GB"; fi
    if [ "$(free_kb)" -lt 1572864 ]; then reason="free memory below 1.5 GB"; fi
    if [ $(( $(swap_mb) - swap_start )) -gt $((max_swap_gb * 1024)) ]; then reason="swap grew more than ${max_swap_gb} GB"; fi
    if [ -n "$reason" ]; then
        # Kill the whole tree: worker processes a job starts itself survive a kill of the root alone.
        tree=$(tree_pids "$pid")
        kill -TERM $tree 2>/dev/null
        for _ in 1 2 3 4 5; do kill -0 $tree 2>/dev/null || break; sleep 2; done
        kill -KILL $tree 2>/dev/null
        break
    fi
    sleep 2
done
wait "$pid" 2>/dev/null; status=$?
tail -n 40 "$log_file"
echo "[guard] exit=$status peak_rss_gb=$(awk -v k="$peak" 'BEGIN {printf "%.2f", k/1048576}') ${reason:+stopped: $reason} log=$log_file"
# Exit with the command's status, or 137 when the guard stopped it.
[ -n "$reason" ] && exit 137
exit "$status"

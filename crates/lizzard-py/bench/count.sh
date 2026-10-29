#!/usr/bin/env bash
# Machine instructions per unit of work, counted by Valgrind (deterministic,
# unlike timings): runs FUNC from units.py with N1 and N2 units and divides
# the difference, so start-up cancels out.
#   count.sh cpython|interp|jit FUNC N1 N2      e.g. count.sh interp fibs 500 1000
# Build the extension with `maturin develop --release` first.
set -euo pipefail
mode=$1 fn=$2 n1=$3 n2=$4
here="$(cd "$(dirname "$0")" && pwd)"
py="${PYTHON:-python}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
count() {
    printf '# lizzard: module\nfrom units import %s\n%s(%s)\n' "$fn" "$fn" "$1" > "$tmp/m.py"
    case $mode in
        cpython) args=("$tmp/m.py") ;;
        interp) export LIZZARD_JIT=0; args=(-m lizzard "$tmp/m.py") ;;
        jit) export LIZZARD_JIT=1; args=(-m lizzard "$tmp/m.py") ;;
    esac
    PYTHONPATH="$here" valgrind --tool=callgrind --callgrind-out-file="$tmp/cg.out" "$py" "${args[@]}" > /dev/null 2> "$tmp/vg.err"
    grep -o "Collected : [0-9]*" "$tmp/vg.err" | grep -o "[0-9]*"
}
a=$(count "$n1")
b=$(count "$n2")
echo "$mode $fn: $(( (b - a) / (n2 - n1) )) instructions per unit"

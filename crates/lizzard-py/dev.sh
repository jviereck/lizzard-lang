#!/usr/bin/env bash
# Build lizzard's Python extension into two venvs (free-threaded 3.14t and the
# GIL 3.14 build) and run the Python tests in each.
#   crates/lizzard-py/dev.sh            build and test both
#   crates/lizzard-py/dev.sh ft         only the free-threaded venv
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
which=("${@:-ft gil}")
for kind in ${which[@]}; do
    venv="$root/.venv-$kind"
    py=$([ "$kind" = ft ] && echo 3.14t || echo 3.14+gil)
    if [ ! -x "$venv/bin/python" ]; then
        uv venv -q --python "$py" "$venv"
        VIRTUAL_ENV="$venv" uv pip install -q numpy maturin
    fi
    echo "== $kind: $("$venv/bin/python" -c 'import sys; print(sys.version.split()[0], "free-threaded" if not sys._is_gil_enabled() else "GIL")')"
    (cd crates/lizzard-py && unset CONDA_PREFIX && VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" maturin develop -q --uv >/dev/null)
    "$venv/bin/python" -m unittest discover -s crates/lizzard-py/tests -t crates/lizzard-py/tests "${UNITTEST_ARGS:--q}"
done

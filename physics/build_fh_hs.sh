#!/usr/bin/env bash
# Build the pinned legacy FH/HB/HS libraries and reconstructed Fortran driver.
set -euo pipefail
project_dir=$(cd "$(dirname "$0")/.." && pwd)
workspace_dir=$(cd "$project_dir/.." && pwd)
# Usage: build_fh_hs.sh [SOURCE_ROOT [TOOLCHAIN_BIN]]
# SOURCE_ROOT contains the three extracted, pinned source directories.
# MSSM_WORK_DIR preserves the local research layout; otherwise use project/vendor.
if [[ -n "${MSSM_WORK_DIR:-}" ]]; then
  work_dir="$MSSM_WORK_DIR"
  source_default="$work_dir/physics_sources"
  compiler_default="$work_dir/toolchain/env/bin"
else
  work_dir="$project_dir/build"
  source_default="$project_dir/vendor"
  compiler_default=""
fi
source_dir=${1:-${SOURCE_ROOT:-"$source_default"}}
compiler_dir=${2:-${TOOLCHAIN_BIN:-"$compiler_default"}}
if [[ -n "$compiler_dir" ]]; then export PATH="$compiler_dir:$PATH"; fi
if [[ "$(uname -s)" == Darwin ]]; then
  if [[ -d /Library/Developer/CommandLineTools ]]; then
    export DEVELOPER_DIR=${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}
  fi
  export SDKROOT=${SDKROOT:-$(xcrun --show-sdk-path)}
  export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}
fi
fc=${FC:-$(command -v gfortran)}
cc=${CC:-$(command -v cc)}
make_bin=${MAKE:-$(command -v make)}
python_bin=${REPRO_PYTHON:-python3}
source_dir=$(cd "$source_dir" && pwd)
mkdir -p "$work_dir"
work_dir=$(cd "$work_dir" && pwd)
log_dir="$work_dir/build_logs"
fh_dir="$source_dir/FeynHiggs-2.13.0"
hb_dir="$source_dir/HiggsBounds-4.3.1"
hs_dir="$source_dir/HiggsSignals-1.4.0"
fh_prefix="$work_dir/install/FeynHiggs-2.13.0"
mkdir -p "$log_dir" "$project_dir/physics/bin" "$work_dir/install"
# Compatibility edits only: explicit BOZ-to-integer conversion and STOP spacing.
"$python_bin" - "$source_dir" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for rel in ('FeynHiggs-2.13.0/src/Main/Couplings.F',
            'FeynHiggs-2.13.0/src/Main/HiggsProd.F'):
    path = root / rel
    text = path.read_text()
    if "int(O'155')" not in text:
        text = text.replace("O'155'", "int(O'155')")
        path.write_text(text)
path = root / 'HiggsSignals-1.4.0/HiggsSignals.F90'
text = path.read_text()
if 'stop"' in text:
    path.write_text(text.replace('stop"', 'stop "'))
PY
(
 cd "$fh_dir"
 ./configure --prefix="$fh_prefix" FC="$fc" CC="$cc" \
   FFLAGS='-O2 -fallow-argument-mismatch -fallow-invalid-boz' > "$log_dir/fh-configure.log" 2>&1
 "$make_bin" -j4 > "$log_dir/fh-make.log" 2>&1
 "$make_bin" install > "$log_dir/fh-install.log" 2>&1
)
(
 cd "$hb_dir"
 ./configure > "$log_dir/hb-configure.log" 2>&1
 "$make_bin" F90C="$fc" F77C="$fc" \
 F90FLAGS='-O2 -fbounds-check -ffixed-line-length-none -fallow-argument-mismatch' \
 > "$log_dir/hb-make.log" 2>&1
)
(
 cd "$hs_dir"
 ./configure > "$log_dir/hs-configure.log" 2>&1
 "$make_bin" F90C="$fc" F77C="$fc" HBLIBS="-L$hb_dir" HBINCLUDE="-I$hb_dir" \
 F90FLAGS='-O2 -fbounds-check -ffixed-line-length-none -fallow-argument-mismatch' \
 > "$log_dir/hs-make.log" 2>&1
)
fh_lib="$fh_prefix/lib"
if [[ -d "$fh_prefix/lib64" ]]; then fh_lib="$fh_prefix/lib64"; fi
"$fc" -O2 -ffixed-line-length-none -fallow-argument-mismatch \
 -I"$fh_prefix/include" -I"$hb_dir" -I"$hs_dir" \
 "$project_dir/physics/fh_hs_driver.F" \
 -L"$fh_lib" -lFH -L"$hs_dir" -lHS -L"$hb_dir" -lHB \
 -o "$project_dir/physics/bin/fh_hs_driver" > "$log_dir/driver-build.log" 2>&1
printf 'Built %s\n' "$project_dir/physics/bin/fh_hs_driver"

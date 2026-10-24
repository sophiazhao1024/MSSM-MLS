#!/bin/sh
# Build the external-SLHA driver against a verified-source manifest version.
# Usage: build_mo.sh SOURCE_ROOT TOOLCHAIN_BIN VERSION
# The source must be a writable extracted copy. No downloads occur here.
set -eu
if [ "$#" -ne 3 ]; then
  echo 'Usage: build_mo.sh SOURCE_ROOT TOOLCHAIN_BIN VERSION (4.3.2 or 4.3.5)' >&2
  exit 64
fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MO_SOURCE=$(CDPATH= cd -- "$1" && pwd)
if [ -n "$2" ]; then
  MO_TOOLCHAIN=$(CDPATH= cd -- "$2" && pwd)
  export PATH="$MO_TOOLCHAIN:$PATH"
fi
MO_VERSION=$3
case "$MO_VERSION" in
  4.3.2) MO_PAPER_MATCH=true; MO_ROBUST_FAST=0 ;;
  4.3.5) MO_PAPER_MATCH=false; MO_ROBUST_FAST=1 ;;
  *) echo 'Only inspected APIs for 4.3.2 and 4.3.5 are supported.' >&2; exit 65 ;;
esac
: "${CC:=cc}"
: "${CXX:=c++}"
: "${FC:=gfortran}"
MO_DEFAULT_CFLAGS='-O2 -fsigned-char -std=gnu99 -fcommon -Wno-error=implicit-function-declaration'
case "$($CC --version 2>/dev/null)" in
  *clang*) MO_DEFAULT_CFLAGS="$MO_DEFAULT_CFLAGS -Wno-error=incompatible-function-pointer-types" ;;
esac
: "${CFLAGS:=$MO_DEFAULT_CFLAGS}"
: "${CXXFLAGS:=-O2 -fcommon}"
: "${FFLAGS:=-O2 -std=legacy -fallow-argument-mismatch}"
: "${MAKE:=make}"
case "$(uname -s)" in
  Darwin)
    : "${DEVELOPER_DIR:=/Library/Developer/CommandLineTools}"
    export DEVELOPER_DIR
    : "${SDKROOT:=$(xcrun --show-sdk-path)}"
    export SDKROOT
    MO_SHARED=-dynamiclib
    MO_SONAME='-install_name '
    MO_RANLIB='ranlib -c'
    MO_FORT_SUFFIX=dylib
    ;;
  Linux) MO_SHARED=-shared; MO_SONAME=''; MO_RANLIB=ranlib; MO_FORT_SUFFIX=so ;;
  *) echo 'Unsupported host: set up compiler flags manually.' >&2; exit 65 ;;
esac
MO_FORT_LIB=$($FC -print-file-name=libgfortran.$MO_FORT_SUFFIX)
case "$MO_FORT_LIB" in */*) MO_FORT_FLAGS="-L$(dirname "$MO_FORT_LIB") -lgfortran";; *) MO_FORT_FLAGS=-lgfortran;; esac
# These are the compiler-selection fields documented by CalcHEP/getFlags.
# No change is made to physical model expressions or integration routines.
cat > "$MO_SOURCE/CalcHEP_src/FlagsForSh" <<EOF
CC="$CC"
CFLAGS="$CFLAGS"
HX11=""
LX11=""
lDL="-ldl"
SHARED="$MO_SHARED"
SONAME="$MO_SONAME"
SO="so"
SNUM="-DNUM_DOUBLE"
FC="$FC"
FFLAGS="$FFLAGS"
lFort="$MO_FORT_FLAGS"
CXX="$CXX"
CXXFLAGS="$CXXFLAGS"
RANLIB="$MO_RANLIB"
MAKE="$MAKE"
export CC CFLAGS lDL LX11 SHARED SONAME SO FC FFLAGS RANLIB CXX CXXFLAGS lFort MAKE
EOF
cd "$MO_SOURCE"
# Sequential archive-member rules prevent parallel ar from losing objects.
"$MAKE" -j1
cp "$SCRIPT_DIR/../physics/mo_driver.c" MSSM/omega_driver.c
cat > MSSM/mo_build_version.h <<EOF
/* Generated from the explicitly selected source manifest version. */
#define MO_SOURCE_VERSION "$MO_VERSION"
#define MO_PAPER_VERSION_MATCH "$MO_PAPER_MATCH"
#define MO_HAS_ROBUST_FAST $MO_ROBUST_FAST
EOF
cd MSSM
# FC= disables the unrelated SuSpect executable target. This driver imports
# an external SLHA spectrum and must never replace it with an internal one.
"$MAKE" -j1 main=omega_driver.c FC=
printf '\nBuilt micrOMEGAs %s driver: %s/MSSM/omega_driver\n' "$MO_VERSION" "$MO_SOURCE"

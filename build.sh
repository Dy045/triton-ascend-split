#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
	# Use project venv when available.
	source "${ROOT_DIR}/.venv/bin/activate"
fi

LLVM_SYSPATH="${LLVM_SYSPATH:-/home/dingyi/llvm-install}"

install_args=(install)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
	# Avoid system site-packages permission issues when no venv is active.
	install_args+=(--user)
fi

cd "${ROOT_DIR}/python"
LLVM_SYSPATH="${LLVM_SYSPATH}" \
TRITON_BUILD_WITH_CCACHE="${TRITON_BUILD_WITH_CCACHE:-true}" \
TRITON_BUILD_WITH_CLANG_LLD="${TRITON_BUILD_WITH_CLANG_LLD:-true}" \
TRITON_BUILD_PROTON="${TRITON_BUILD_PROTON:-OFF}" \
DEBUG="${DEBUG:-1}" \
TRITON_WHEEL_NAME="${TRITON_WHEEL_NAME:-triton-ascend}" \
TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:--DTRITON_BUILD_UT=OFF}" \
python3 setup.py "${install_args[@]}"

build_dir=""
for d in build/cmake.*; do
	if [[ -d "${d}" && -f "${d}/compile_commands.json" ]]; then
		build_dir="${d}"
		break
	fi
done

if [[ -z "${build_dir}" ]]; then
	echo "ERROR: compile_commands.json not found under python/build/cmake.*" >&2
	exit 1
fi

ln -sf "${ROOT_DIR}/python/${build_dir}/compile_commands.json" "${ROOT_DIR}/compile_commands.json"
echo "Linked compile_commands.json from ${build_dir}"

#!/bin/bash
# CANN OHOS device helpers — push and test the Rust device test binary.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "$CANN_DDK" ]; then
    echo "Error: CANN_DDK is not set."
    echo "  export CANN_DDK=/path/to/CANN-Kit-next/ddk/"
    exit 1
fi

DDK_LIB="${CANN_DDK}/ai_ddk_lib/lib64"

BINARY_DIR="${PROJECT_DIR}/target/aarch64-unknown-linux-ohos/release/deps"
# libtest gives the test binary a hashed name; pick the most recently built.
BINARY="$(ls -t "${BINARY_DIR}"/test_cann_execution-* 2>/dev/null | head -1)"
DEVICE_BIN="test_cann_execution"

# ── Helpers ────────────────────────────────────────────────────────────

ok()  { echo "  [OK] $*"; }
fail(){ echo "  [FAIL] $*"; exit 1; }

# ── Push to device ─────────────────────────────────────────────────────

push_to_device() {
    local target="${1:-/data/local/tmp/cann-test}"

    if ! command -v hdc &>/dev/null; then
        echo "Error: hdc not found. Install Huawei DevEco Device Tool."
        exit 1
    fi

    echo "=== Pushing to ${target} ==="
    hdc shell "mkdir -p ${target}"

    if [ -n "$BINARY" ] && [ -f "$BINARY" ]; then
        echo "  ${BINARY##*/} -> ${target}/${DEVICE_BIN}"
        hdc file send "$BINARY" "${target}/${DEVICE_BIN}"
    else
        echo "  WARN: test_cann_execution binary not found. Run 'make cann-device-test' first."
    fi

    for lib in libhiai.so libhiai_ir.so libhiai_ir_build.so libhiai_ir_build_aipp.so; do
        local src="${DDK_LIB}/${lib}"
        if [ -f "$src" ]; then
            echo "  ${lib} -> ${target}/"
            hdc file send "$src" "${target}/"
        fi
    done

    hdc shell "chmod +x ${target}/${DEVICE_BIN}"
    ok "pushed"
}

# ── Test on device ─────────────────────────────────────────────────────

test_on_device() {
    local target="${1:-/data/local/tmp/cann-test}"
    push_to_device "$target"
    echo ""
    echo "=== Running on device ==="
    hdc shell "cd ${target} && LD_LIBRARY_PATH=. ./${DEVICE_BIN} --nocapture --test-threads=1"
}

# ── WPT conformance on device ──────────────────────────────────────────

find_wpt_binary() {
    local bin
    bin=$(ls -t "${PROJECT_DIR}"/target/aarch64-unknown-linux-ohos/release/deps/run_wpt_conformance-* 2>/dev/null \
        | grep -v '\.d$' | head -n1)
    if [ -z "$bin" ] || [ ! -f "$bin" ]; then
        fail "run_wpt_conformance test binary not found; run 'make test-wpt-cann' first"
    fi
    echo "$bin"
}

dump_wpt_corpus() {
    local wpt_cache="${WPT_DIR:-${PROJECT_DIR}/.cache/wpt}"
    local out="${1:-/tmp/wpt-corpus.json}"
    if [ ! -d "${wpt_cache}/webnn/conformance_tests" ]; then
        echo "  [INFO] WPT cache missing; fetching (node scripts/fetch_wpt.mjs)..."
        (cd "${PROJECT_DIR}" && node scripts/fetch_wpt.mjs)
    fi
    echo "  [INFO] Dumping WPT corpus -> ${out}"
    (cd "${PROJECT_DIR}" && node scripts/wpt_bridge/dump_corpus.mjs --wpt-dir "${wpt_cache}" > "${out}")
    ok "corpus dumped"
}

wpt_on_device() {
    local target="${1:-/data/local/tmp/cann-wpt}"
    local wpt_bin
    local corpus_json="/tmp/wpt-corpus.json"
    wpt_bin=$(find_wpt_binary)

    if ! command -v hdc &>/dev/null; then
        echo "Error: hdc not found. Install Huawei DevEco Device Tool."
        exit 1
    fi

    dump_wpt_corpus "$corpus_json"

    echo "=== Pushing WPT to ${target} ==="
    hdc shell "mkdir -p ${target}"

    local bin_name
    bin_name=$(basename "$wpt_bin")
    echo "  ${bin_name} -> ${target}/"
    hdc file send "$wpt_bin" "${target}/"
    hdc shell "chmod +x ${target}/${bin_name}"

    echo "  wpt-corpus.json -> ${target}/"
    hdc file send "$corpus_json" "${target}/"

    for lib in libhiai.so libhiai_ir.so libhiai_ir_build.so libhiai_ir_build_aipp.so; do
        local src="${DDK_LIB}/${lib}"
        if [ -f "$src" ]; then
            echo "  ${lib} -> ${target}/"
            hdc file send "$src" "${target}/"
        fi
    done
    ok "pushed"

    echo ""
    echo "=== Running WPT on device (backend=cann) ==="
    hdc shell "cd ${target} && LD_LIBRARY_PATH=. WPT_CORPUS_JSON=./wpt-corpus.json WPT_BACKEND=cann ./${bin_name}"

    echo ""
    echo "  [INFO] report disabled; set WPT_REPORT_JSON=./reports/wpt-conformance.json and"
    echo "         WPT_REPORT_HTML= in the run above to emit a rustnnpt-compatible report."
}

# ── Main ───────────────────────────────────────────────────────────────

TARGET="${1:-test}"

case "$TARGET" in
    push) push_to_device "$2" ;;
    test) test_on_device "$2" ;;
    wpt) wpt_on_device "$2" ;;
    *)
        echo "Usage: $0 [push|test|wpt] [target-dir]"
        echo ""
        echo "  push    Transfer files to OHOS device via hdc"
        echo "  test    Push + execute test_cann_execution on device"
        echo "  wpt     Push + execute WPT conformance (backend=cann) on device"
        echo ""
        echo "  Default target-dir: /data/local/tmp/cann-test"
        exit 1
        ;;
esac

#!/usr/bin/env bash
#
# parallel-rsync.sh — copy a tree in parallel with N rsync jobs
# Example:
#   ./parallel-rsync.sh -j 12 -s conf/ -d runpod:/workspace/conf -- --delete
#
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [-j JOBS] -s SRC -d DST [-- RSYNC_EXTRA_ARGS]

Options
  -j JOBS   Number of parallel rsync jobs (default: 8)
  -s SRC    Source directory or files (required)
  -d DST    Destination (required)
  -h        Show this help and exit

Anything after a bare “--” is passed straight through to each rsync invocation.
EOF
}

# ---------- Parse options ----------
JOBS=8
SRC=""
DST=""

while getopts ":j:s:d:h" opt; do
  case "$opt" in
    j) JOBS="$OPTARG" ;;
    s) SRC="$OPTARG" ;;
    d) DST="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Everything after “--” goes to rsync unchanged
RSYNC_EXTRA=()
if [[ ${1:-} == "--" ]]; then
  shift
  RSYNC_EXTRA=("$@")
fi

# ---------- Sanity checks ----------
[[ -z $SRC ]] && { echo "Error: -s SRC is required"; usage; exit 1; }
[[ -z $DST ]] && { echo "Error: -d DST is required"; usage; exit 1; }
[[ ! -e $SRC ]] && { echo "Error: source '$SRC' does not exist"; exit 1; }

# ---------- Work files ----------
LIST_FILE=$(mktemp /tmp/rsync.list.XXXXXX)
CHUNK_PREFIX=/tmp/rsync_chunk_

cleanup() { rm -f "$LIST_FILE" ${CHUNK_PREFIX}*; }
trap cleanup EXIT

# 1. Generate authoritative list of files to copy
rsync -azn --copy-links --out-format='%n' "$SRC" "$DST" > "$LIST_FILE"

# 2. Split list into N chunks (l = line-balanced split)
split -n l/"$JOBS" "$LIST_FILE" "$CHUNK_PREFIX"

# 3. Run parallel rsyncs
parallel -j "$JOBS" \
  rsync -aP --no-g --no-o --copy-links --files-from={} \
        "${RSYNC_EXTRA[@]}" "$SRC" "$DST" \
  ::: ${CHUNK_PREFIX}*

echo "✅  Copy complete with $JOBS parallel job(s)."

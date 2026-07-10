#!/usr/bin/env bash
set -euo pipefail

BINARY="${1:-./build/l4rpz}"
PASS=0
FAIL=0

check() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        FAIL=$((FAIL + 1))
    fi
}

# check that binary exits with a specific code
check_exit() {
    local desc="$1" expected_code="$2"
    shift 2
    local actual_code=0
    "$@" > /dev/null 2>&1 || actual_code=$?
    check "$desc" "$expected_code" "$actual_code"
}

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# ── header checksum ───────────────────────────────────────────────────────────

printf 'HELLOWORLD HELLOWORLD' > "$tmpdir/base.raw"
lz4 -q "$tmpdir/base.raw" "$tmpdir/base.lz4"

# valid file works
check_exit "header checksum: valid file accepted" 0 \
    "$BINARY" "HELLOWORLD" "$tmpdir/base.lz4"

# flip the HC byte (offset 6 for standard files: magic=4, FLG=1, BD=1, then HC)
python3 -c "
data = bytearray(open('$tmpdir/base.lz4','rb').read())
data[6] ^= 0xFF
open('$tmpdir/bad_hc.lz4','wb').write(data)
"
check_exit "header checksum: corrupted HC byte rejected" 1 \
    "$BINARY" "HELLOWORLD" "$tmpdir/bad_hc.lz4"

# flip a non-HC byte in the header (FLG byte = offset 4) — also changes HC validity
python3 -c "
data = bytearray(open('$tmpdir/base.lz4','rb').read())
data[4] ^= 0x01  # flip reserved bit in FLG
open('$tmpdir/bad_flg.lz4','wb').write(data)
"
check_exit "header checksum: FLG tampered (HC now wrong) rejected" 1 \
    "$BINARY" "HELLOWORLD" "$tmpdir/bad_flg.lz4"

# ── block checksum (-BX) ──────────────────────────────────────────────────────

lz4 -q -BX "$tmpdir/base.raw" "$tmpdir/bx.lz4"

# valid file with block checksum works
check_exit "block checksum: valid file accepted" 0 \
    "$BINARY" "HELLOWORLD" "$tmpdir/bx.lz4"

# locate and corrupt the block checksum bytes
python3 -c "
data = bytearray(open('$tmpdir/bx.lz4','rb').read())
# header: magic(4) + FLG(1) + BD(1) + HC(1) = 7 bytes
# block:  size_field(4) + block_data(N)
pos = 7
block_size_field = int.from_bytes(data[pos:pos+4], 'little')
pos += 4
is_compressed = not (block_size_field >> 31)
block_data_size = block_size_field & ~(1 << 31)
pos += block_data_size
# block checksum is here (4 bytes)
print(f'block checksum at offset {pos}: {data[pos:pos+4].hex()}')
data[pos] ^= 0xFF  # corrupt first byte of block checksum
open('$tmpdir/bad_bx.lz4','wb').write(data)
"
check_exit "block checksum: corrupted block checksum rejected" 1 \
    "$BINARY" "HELLOWORLD" "$tmpdir/bad_bx.lz4"

# multi-block file with block checksums — corrupt checksum of second block
python3 -c "
import sys
sys.stdout.buffer.write(b'A' * (64*1024) + b'HELLOWORLD' + b'B' * (64*1024 - 10))
" > "$tmpdir/big.raw"
lz4 -q -B4 -BX "$tmpdir/big.raw" "$tmpdir/big_bx.lz4"

check_exit "block checksum: multi-block valid file accepted" 0 \
    "$BINARY" "HELLOWORLD" "$tmpdir/big_bx.lz4"

python3 -c "
data = bytearray(open('$tmpdir/big_bx.lz4','rb').read())
# skip to second block's checksum
pos = 7
for _ in range(2):
    block_size_field = int.from_bytes(data[pos:pos+4], 'little')
    pos += 4
    block_data_size = block_size_field & ~(1 << 31)
    pos += block_data_size
    pos += 4  # skip past this block's checksum
# now corrupt the second block checksum (went one too far, back up)
pos -= 4
print(f'second block checksum at offset {pos}')
data[pos] ^= 0xFF
open('$tmpdir/big_bad_bx.lz4','wb').write(data)
"
check_exit "block checksum: corrupted second block checksum rejected" 1 \
    "$BINARY" "HELLOWORLD" "$tmpdir/big_bad_bx.lz4"

# ── content checksum (not yet verified — should still work) ───────────────────

# content checksum is on by default in lz4 CLI; corrupt it and confirm
# the tool currently does NOT verify it (known TODO)
python3 -c "
data = bytearray(open('$tmpdir/base.lz4','rb').read())
data[-1] ^= 0xFF  # last 4 bytes are the content checksum
open('$tmpdir/bad_cc.lz4','wb').write(data)
"
check_exit "content checksum: corrupted (not yet verified, should pass)" 0 \
    "$BINARY" "HELLOWORLD" "$tmpdir/bad_cc.lz4"

echo ""
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

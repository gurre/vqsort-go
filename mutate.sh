#!/bin/bash
# Checks that the test suite actually catches the bugs it is supposed to catch.
#
# A passing suite proves nothing on its own: the interesting failures of an
# in-place vector partition are silent ones, where the output is still sorted or
# still plausible. Each mutation below is a real bug this kernel can have. Every
# one of them must make the suite fail; a mutation that survives means the suite
# has a hole, not that the code is fine.
#
# Usage: ./mutate.sh
set -u

cd "$(dirname "$0")"
export GOEXPERIMENT=simd

# name | file | search | replace
mutations=(
"perm table: two lanes swapped|tables_neon.go|permTable32x4[m][4*j+b] = lane*4 + uint8(b)|permTable32x4[m][4*j+b] = ((lane+1)%4)*4 + uint8(b)"
"perm table: index out of range|tables_neon.go|permTable64x2[m][8*j+b] = lane*8 + uint8(b)|permTable64x2[m][8*j+b] = lane*8 + uint8(b) + 16"
"remainder spill counted one short|kernel_i32_neon.go|copy(keys[wl:wr], buf[:nBuf])|copy(keys[wl:wr], buf[:max(nBuf-1, 0)])"
"drain uses the overlapping store|kernel_i32_neon.go|wl, wr = storeLastI32(keys, wl, wr, pv, r3)|wl, wr = storeOneI32(keys, wl, wr, pv, r3)"
"partition compares the wrong way|kernel_i32_neon.go|right := pivot.Less(v) // lanes strictly greater than the pivot|right := pivot.LessEqual(v) // lanes strictly greater than the pivot"
"lane count off by one|kernel_i32_neon.go|return lr, lanes32 - bits.OnesCount32(m)|return lr, lanes32 - bits.OnesCount32(m) - 1"
"depth guard never trips|order.go|return 2*d + 4|return 1 << 30"
"named types lose their kernel|sort.go|case reflect.Int64:|case reflect.Invalid:"
"network comparator inverted|network_i32_neon.go|v0, v1 = minKeyI32(v0, v1), maxKeyI32(v0, v1)|v0, v1 = maxKeyI32(v0, v1), minKeyI32(v0, v1)"
"network direction mask wrong|network_neon.go|dirK16Bits = [4]int32{-1, 0, -1, 0}|dirK16Bits = [4]int32{-1, -1, 0, 0}"
"network lane shuffle wrong|network_neon.go|laneSwap1Bytes = [16]uint8{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11}|laneSwap1Bytes = [16]uint8{0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11}"
"64-bit min and max swapped|network_neon.go|func minKeyI64(a, b archsimd.Int64x2) archsimd.Int64x2 { return b.IfElse(a.Greater(b), a) }|func minKeyI64(a, b archsimd.Int64x2) archsimd.Int64x2 { return a.IfElse(a.Greater(b), b) }"
"base case padding does not sort last|network_neon.go|const padI32 int32 = math.MaxInt32|const padI32 int32 = 0"
)

fail=0
for m in "${mutations[@]}"; do
	IFS='|' read -r name file search replace <<<"$m"
	cp "$file" "$file.orig"
	python3 - "$file" "$search" "$replace" <<'PY'
import sys
path, search, replace = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if search not in s:
	sys.exit("mutation target not found in %s: %s" % (path, search))
open(path, 'w').write(s.replace(search, replace, 1))
PY
	if [ $? -ne 0 ]; then
		mv "$file.orig" "$file"
		echo "SKIP  $name (target text has moved)"
		fail=1
		continue
	fi

	if go test -short -count=1 ./... >/dev/null 2>&1; then
		echo "ALIVE $name -- the suite did not catch this"
		fail=1
	else
		echo "ok    $name"
	fi
	mv "$file.orig" "$file"
done

if [ "$fail" -ne 0 ]; then
	echo
	echo "at least one mutation survived; the suite has a gap"
	exit 1
fi
echo
echo "all mutations caught"

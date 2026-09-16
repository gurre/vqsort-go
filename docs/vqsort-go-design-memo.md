# vqsort in Go: technical design memo

Port of Blacher, Giesen, Wassenberg, Sanders, *Vectorized and performance-portable Quicksort* (arXiv:2205.05982, the Highway `vqsort` implementation) to pure Go on top of `simd/archsimd`.

## 1. Decision and scope

Implement vqsort as a Go package with hand-written SIMD kernels using `simd/archsimd` (Go 1.27, `GOEXPERIMENT=simd`), a scalar fallback, and runtime CPU dispatch. No assembly, no cgo.

In scope for v1: ascending sort of `[]int32`, `[]uint32`, `[]float32`, `[]int64`, `[]uint64`, `[]float64` on amd64 with AVX2 and AVX-512 kernels. Semantics match `slices.Sort` for these element types, including NaN ordering.

Out of scope for v1: 16-bit keys (compress needs VBMI2, AVX2 has no 16-bit masked store), 128-bit keys, descending order, custom comparators, arm64, parallel sorting. Sections 15 and 17 record where each of these plugs in.

The algorithm is taken from the paper unchanged: bidirectional in-place partition with compress, cache-line chunk pivot sampling with median-of-3 reduction, a 16-row transpose-free sorting network base case, a recursion depth guard with heapsort fallback. The engineering content of this memo is the mapping to Go's compiler and to the archsimd API, and the places where Go forces a different construction than Highway uses.

## 2. Toolchain prerequisites

- Go 1.27.x, built with `GOEXPERIMENT=simd`. Pin with `toolchain go1.27.1` in `go.mod`. The archsimd API broke between 1.26 and 1.27 and is documented as unstable; expect another break at 1.28 when the proposal to enable amd64 archsimd by default lands (golang/go#78979).
- Kernel files carry `//go:build goexperiment.simd && amd64`. Everything else in the package builds without the experiment.
- Reinterpretation between element types uses `ToBits()`, `BitsToInt64()`, `BitsToFloat64()`, `ReshapeToUint32s()`, `ReshapeToUint64s()`. The `As*` methods are marked deprecated in 1.27 and are not used.

## 3. Package layout and API

```
vqsort/
  sort.go            public API, generic wrapper with type switch
  dispatch_amd64.go  kernel selection at init, test override
  fallback.go        //go:build !(goexperiment.simd && amd64): slices.Sort
  scalar.go          insertion sort, heapsort, NaN prepass, SFC64, remainder partition
  gen/               text/template kernel generator, run by go generate
  kernel_i64_avx512.go   generated
  kernel_i64_avx2.go     generated
  ...                    one file per (key type, target)
  tables.go          permutation tables, built at init
```

Public API:

```go
package vqsort

type Key interface {
	~int32 | ~uint32 | ~float32 | ~int64 | ~uint64 | ~float64
}

// Sort sorts s in ascending order. Floating-point NaNs are ordered first,
// matching slices.Sort.
func Sort[T Key](s []T)
```

`Sort` type-switches on `any(s)` to the concrete kernel entry point. Generics do not reach into the kernels: GC-shape stenciling cannot select `Int64x8` versus `Float64x8` from a type parameter, so every kernel is concrete and generated (section 5). Named types with the listed underlying types are converted with `unsafe.Slice` in the wrapper.

## 4. Dispatch and fallbacks

```go
var sortInt64 func([]int64) = sortInt64Scalar

func init() {
	switch {
	case archsimd.X86.AVX512():
		sortInt64 = sortInt64AVX512
	case archsimd.X86.AVX2():
		sortInt64 = sortInt64AVX2
	}
}
```

One function variable per key type, assigned once. The indirect call is per `Sort` invocation, not per partition step.

AVX2 is the floor. No 128-bit kernel ships: archsimd emulated ops at 128 bits have been found to use AVX2 instructions on AVX-only CPUs (golang/go#81405), and 2 to 4 lanes give little over `slices.Sort`.

A test override `vqsort.SetTarget(t Target)` with `Target ∈ {Scalar, AVX2, AVX512}` forces a kernel regardless of CPU, guarded so that forcing AVX-512 on a machine without it panics with a message rather than SIGILL. It exists so one CI machine exercises every kernel.

On non-amd64 or without the experiment, `fallback.go` defines every `sort*` symbol as a call to `slices.Sort`.

## 5. Kernel generation

One template, instantiated per (key type, target). Template parameters:

| parameter | i64 / AVX-512 | i64 / AVX2 | i32 / AVX2 | i32 / AVX-512 |
|---|---|---|---|---|
| `T` | `int64` | `int64` | `int32` | `int32` |
| `V` | `Int64x8` | `Int64x4` | `Int32x8` | `Int32x16` |
| `M` | `Mask64x8` | `Mask64x4` | `Mask32x8` | `Mask32x16` |
| `N` | 8 | 4 | 8 | 16 |
| `Bits` | `uint8` | `uint8` | `uint8` | `uint16` |
| `partitionVec` variant | table-permute-8 | table-permute-4 | table-permute-8 | compress-expand |
| `Last` (padding sentinel) | `MaxInt64` | `MaxInt64` | `MaxInt32` | `MaxInt32` |

Float variants set `Last` to `+Inf` and use the float vector types; unsigned variants set `Last` to the max value. This is the same shape as archsimd's own `tmplgen`. The generator is checked in, the generated files are checked in, and CI verifies `go generate` produces no diff.

The `partitionVec` variant is the only structural difference between kernels (section 7.1). Everything else varies only by type names and `N`.

## 6. Buffers and stack discipline

Each top-level `sort*` allocates one buffer on its stack and passes a pointer down the recursion:

```go
const bufLenI64AVX512 = 256 + 2*8   // NBaseCase + two vectors of padding
var buf [bufLenI64AVX512]int64
```

Size is `max(256 + 2N, 9N, 4·chunkLanes + 2N)` per the paper's constraints; `256 + 2N` dominates for every v1 instantiation. The buffer is used by the base case (copy in, pad, sort, copy out), by the remainder loop of partition (right-side spill), and by pivot sampling (median scratch).

Vectors are always values in registers or locals. No vector is stored in a struct, slice, or heap object; archsimd documents that this defeats register allocation. Permutation tables are `[][16]uint8` arrays, loaded into vectors at the point of use.

## 7. Partition

### 7.1 Vector partition primitive

Highway's AVX-512 path uses `CompressStore` (writes exactly popcount lanes) for the left side and `CompressBlendedStore` for the right side. Go has no masked store on a slice, only `StoreArrayMasked(*[N]T, mask)`, and the array pointer conversion requires `N` in-bounds elements. The right-side store near the end of the slice would need `wr - nRight + N <= len`, which fails on the first right-side store. Rather than reproduce Highway's "make space for one vector" epilogue, every kernel uses full-width stores of a vector that holds left lanes at the bottom and right lanes at the top. This is exactly Highway's non-native (AVX2) path, and it is correct on AVX-512 as well. The primitive:

```go
// partitionVec returns v with lanes <= pivot packed at the low end and the
// remaining lanes packed at the high end, relative order preserved within
// each group, plus the count of low lanes.
func partitionVec(v, pivot V) (lr V, numLeft int)
```

Three implementations:

**Table-permute-8** (`Int64x8` on AVX-512, `Int32x8` on AVX2). A 256-entry table indexed by the right-mask bits gives the 8 lane indices, left lanes first. Entries are 8 bytes, padded to 16 for a single unmasked load, widened with `VPMOVZXB{D,Q}`.

```go
var permTable8 [256][16]uint8 // [m]: indices of clear bits, then set bits

func partitionVec(v, pivot archsimd.Int64x8) (archsimd.Int64x8, int) {
	right := pivot.Less(v)                         // VPCMPQ
	m := right.ToBits()                            // KMOVB
	idx := archsimd.LoadUint8x16Array(&permTable8[m]).ExtendLo8ToUint64() // VPMOVZXBQ
	return v.Permute(idx), 8 - bits.OnesCount8(m)  // VPERMQ zmm
}
```

For `Int32x8` on AVX2 the last two lines become `ExtendLo8ToUint32()` and `Permute` on `Uint32x8` indices (`VPERMD ymm`, native AVX2). Table size 4 KB. `pivot.Less(v)` is used rather than `v.LessEqual(pivot)` because 64-bit `LessEqual` is emulated on AVX2 (compare, then xor with all-ones).

**Table-permute-4** (`Int64x4`, `Uint64x4`, `Float64x4` on AVX2). AVX2 has no variable 64-bit lane permute (`VPERMQ` with vector indices is AVX512VL). Use `VPERMD` with doubled indices: a 16-entry table of 8 `uint32` indices `(2·lane, 2·lane+1)`. Path: `v.ToBits().ReshapeToUint32s().Permute(idx).ReshapeToUint64s().BitsToInt64()`. These reshapes are free (register reinterpretation). Do not call `Int64x4.Permute` on the AVX2 kernel until its CPU-feature annotation is confirmed to be AVX2.

**Compress-expand** (`Int32x16`, `Uint32x16`, `Float32x16` on AVX-512). A 16-lane table has 65536 entries and is not viable. Two compresses and an expand:

```go
func partitionVec(v, pivot archsimd.Int32x16) (archsimd.Int32x16, int) {
	right := pivot.Less(v)
	rb := right.ToBits()                                    // uint16
	nr := bits.OnesCount16(rb)
	left := archsimd.Mask32x16FromBits(^rb)
	nl := 16 - nr
	lo := v.Compress(left)                                                   // VPCOMPRESSD, left lanes at bottom
	hi := v.Compress(right).Expand(archsimd.Mask32x16FromBits(^uint16(0) << nl)) // VPCOMPRESSD + VPEXPANDD, right lanes at top
	keep := archsimd.Mask32x16FromBits(uint16(1)<<nl - 1)
	return lo.IfElse(keep, hi), nl                                           // low nl lanes from lo, rest from hi
}
```

Cost is three p5 shuffle ops per 16 keys against one per 8 keys for the table path. Benchmark alternative: split into `GetLo()`/`GetHi()` `Int32x8` halves and run table-permute-8 on each (two 8-lane store pairs per 16 keys). Decide by measurement (section 14).

### 7.2 Main loop

Precondition: `len(keys) > NBaseCase`. The loop body processes `U = 4N` keys per iteration (the paper's unroll factor; the Go compiler does not unroll, so the four steps are written out).

```go
func partition(keys []int64, pivot archsimd.Int64x8, ps int64, buf *[bufLen]int64) int {
	const N, U = 8, 32
	n := len(keys)
	rem := n % U
	wl, nBuf := partitionRemainder(keys[:rem], ps, buf[:]) // section 7.3
	readL, readR, wr := rem, n, n

	l0, l1, l2, l3 := load4(keys, readL); readL += U
	r0, r1, r2, r3 := load4(keys, readR-U); readR -= U

	for readL < readR {
		var v0, v1, v2, v3 archsimd.Int64x8
		if wr-readR < readL-wl { // less free space on the right: read from the right
			readR -= U
			v0, v1, v2, v3 = load4(keys, readR)
		} else {
			v0, v1, v2, v3 = load4(keys, readL)
			readL += U
		}
		wl, wr = storeLR(keys, wl, wr, v0, pivot)
		wl, wr = storeLR(keys, wl, wr, v1, pivot)
		wl, wr = storeLR(keys, wl, wr, v2, pivot)
		wl, wr = storeLR(keys, wl, wr, v3, pivot)
	}
	// readL == readR: the free region [wl, wr) is now contiguous.
	wl, wr = storeLR(keys, wl, wr, l0, pivot)
	// ... l1, l2, l3, r0, r1, r2, r3
	copy(keys[wl:wr], buf[:nBuf])   // wr - wl == nBuf
	return wl
}

func storeLR(keys []int64, wl, wr int, v, pivot archsimd.Int64x8) (int, int) {
	lr, nl := partitionVec(v, pivot)
	lr.StoreArray((*[8]int64)(keys[wl : wl+8 : wl+8]))
	lr.StoreArray((*[8]int64)(keys[wr-8 : wr : wr]))
	return wl + nl, wr - (8 - nl)
}

func load4(keys []int64, i int) (a, b, c, d archsimd.Int64x8) {
	p := (*[32]int64)(keys[i : i+32 : i+32])
	return archsimd.LoadInt64x8Array((*[8]int64)(p[0:8])), archsimd.LoadInt64x8Array((*[8]int64)(p[8:16])),
		archsimd.LoadInt64x8Array((*[8]int64)(p[16:24])), archsimd.LoadInt64x8Array((*[8]int64)(p[24:32]))
}
```

Invariants that make the full-width stores safe:

1. Free space `capL = readL - wl` on the left and `capR = wr - readR` on the right satisfy `capL + capR = 8N + nBuf` at the top of every iteration (preload of `8N`, plus the remainder gap).
2. The load is taken from the side with less free space. After the load the smaller side has at least `4N`, and the other side had at least half of `8N + nBuf`, so both sides have `>= 4N` before the four stores. Each store consumes at most `N` on one side, so `capL >= N` before every left store and `capR >= N` before every right store. The store windows `[wl, wl+N)` and `[wr-N, wr)` therefore lie inside already-read memory and never touch unread keys.
3. The left store writes `nl` valid keys and `N - nl` garbage lanes above `wl + nl`; the right store writes `N - nl` valid keys at the top and `nl` garbage lanes below `wr - (N - nl)`. All garbage lands in the free region and is overwritten by later stores, because the free region shrinks to exactly `nBuf` and every position in it is eventually claimed by a store whose valid lanes cover it.
4. At loop exit `readL == readR`, so the free region is one contiguous interval and the eight preloaded vectors drain without a capacity check.

Bounds checks: the three-index slice expressions and array pointer conversions concentrate the checks to one per load or store. `load4` takes one check for 32 keys. Verify with `-gcflags=-d=ssa/check_bce` that the inner loop carries no others.

### 7.3 Remainder loop

`rem < 4N` keys at the front of the range are partitioned by a scalar loop: left keys compacted in place at `keys[0:wl]`, right keys appended to `buf`. Highway vectorizes this with blended compress stores; in Go the vector variant needs `StoreArrayMasked` in place (fine, `wl + N <= len` holds since `len > 256`) and a compress to the buffer. This is at most 31 keys per partition call with `N = 8` and is written scalar in v1, branchless (`keys[wl] = k; buf[nBuf] = k; wl += isLeft; nBuf += 1 - isLeft`). Revisit after profiling; the paper's measurement is that it is not on the critical path.

## 8. Pivot selection

Direct port of section 2.2 of the paper.

- RNG: SFC64, scalar, seeded per `Sort` call from a package-level counter (deterministic runs under test, no `math/rand` dependency). Bounded draw by 32-bit multiply-shift (O'Neill), bias accepted as in the paper.
- Nine 64-byte chunks at random chunk-aligned offsets within the range: offset `= rng.bounded(uint32(n/chunkLanes - 1)) * chunkLanes`. For 64-bit keys a chunk is one `Int64x8` on AVX-512 or two `Int64x4` on AVX2; for 32-bit keys one `Int32x16` or two `Int32x8`.
- Per-lane median of three across each triple of chunks: `medianOf3(a,b,c) = a.Min(b).Max(a.Max(b).Min(c))`, four min/max ops. Three triples produce three median vectors, stored to `buf`.
- Reduce: median-of-3 of the three vectors gives one vector of `N` medians. Store it and finish scalar: repeatedly replace groups of three by their median until fewer than three remain, take the first. Scalar work is `O(N)` per partition call.
- The pivot is always a key present in the range. This is required by the degenerate-partition logic in section 10.
- The pivot is passed to `partition` both as a scalar (remainder loop) and as a broadcast vector (`BroadcastInt64x8`).

Chunk offsets are element indices, not byte-aligned addresses; slices carry no 64-byte alignment guarantee and none is needed.

## 9. Base case sorting network

Ranges with `n <= NBaseCase = 16·N` are sorted in registers.

1. Copy `n` keys into `buf` with `LoadInt64x8Part`/`StorePart` for the ragged tail (no read past `len`), then pad `buf[n : roundUp(n, 16·c)]` with `Last` so the padding sorts to the end and is discarded.
2. View the buffer as 16 rows of `c` lanes, `c` the smallest power of two with `16·c >= n` and `c <= N`. In v1 the width selection is `c = N` for `n > 8N`, and scalar insertion sort for `n <= 32`. The intermediate widths (`c = N/2`, `N/4`) are additional template instantiations of the same network over `Int64x4` and `Int64x2`, added in M5 once the full-width path is measured.
3. Column sort: Green's 60-comparator 16-input network (Knuth TAOCP 5.3.4, as coded in Highway's `sorting_networks-inl.h`), each comparator a `Min`/`Max` pair between two row vectors. 16 live vectors plus temporaries. On AVX2 this exceeds the 16 `ymm` registers and will spill; the base case is a few hundred cycles per call and the spill cost is measured, not assumed (section 12).
4. Transpose-free bitonic merge of adjacent columns, then column pairs, up to `c`. Each merge step is a lane permutation of one operand followed by `Min`/`Max` and a fixed-mask blend `IfElse`. The permutations needed are reversal within groups of 2, 4, 8, 16 lanes and adjacent-lane swaps. archsimd ops: `PermuteScalarsGrouped(a,b,c,d)` for 32-bit shuffles inside 128-bit blocks, `ConcatPermuteScalarsGrouped(a,b,y)` for 64-bit, `ConcatPermute128Scalars(lo,hi,y)` to swap 128-bit halves on 256-bit vectors, and `Permute(indices)` with a hoisted constant index vector for the 512-bit cross-block cases. The merge sequences are taken from Highway's `vqsort-inl.h` (`Merge2`, `Merge4`, `Merge8`, `Merge16`) and translated op for op; no new network design is done here.
5. Copy `n` keys back with `StorePart` for the tail.

Padding correctness for floats depends on NaNs being absent (section 11).

## 10. Recursion, degenerate partitions, depth guard

`recurse(keys, pivot, buf, rng, depth)` mirrors Algorithm 1 of the paper:

- `bound := partition(keys, pivot)`; `left = keys[:bound]`, `right = keys[bound:]`. The left side is never empty because the pivot is a key and `<= pivot` goes left.
- If `right` is empty: `scanMinMax` over the range (per-lane `Min`/`Max` accumulation, `GetLo/GetHi` halving to 128 bits, `GetElem`). If `min == max` the range is sorted, return. Otherwise recurse with `pivot = min`, which is guaranteed to split off at least one key.
- Ranges `<= NBaseCase` go to the base case; larger ranges get a fresh `choosePivot` before recursing. Pivot choice is hoisted out of the callee exactly as in the paper so the degenerate case can substitute its own pivot.
- Depth guard: `maxDepth = 2·bits.Len(uint(n)) + 4`. On exceeding it, the range is heapsorted (scalar, `container/heap` is not used; a plain sift-down on the slice). This bounds the worst case at `O(n log n)`.
- Go has no tail-call elimination. Frame size stays small: the buffer is a pointer, the RNG is a pointer, vectors are not kept live across the recursive call (the pivot vector is rebroadcast from its scalar after return). Depth is bounded by the guard, so stack growth is bounded.

## 11. Order abstraction and floating point

Ascending only in v1. The template exposes `Compare`, `First`, `Last`, `FirstOfLanes`, `LastOfLanes` so descending is a second instantiation later (swap `Min`/`Max`, `Less`/`Greater`, `First`/`Last`).

Floats:

- NaN: `slices.Sort` orders NaNs before all other values. `sortFloat64` runs a prepass that moves NaNs to the front (scalar in v1, `IsNaN` mask plus compress later) and sorts `s[nanCount:]`. With NaNs excluded, `Min`/`Max`/`Less` are a total order and `+Inf` is a valid `Last` sentinel.
- Signed zero: `-0.0` and `+0.0` compare equal under `Less`, `Min`, and `Max`; their relative order in the output is unspecified, which matches `slices.Sort`.
- Unsigned 64-bit on AVX2: `Uint64x4.Min`/`Max`/`Less` are emulated (no `VPMINUQ` below AVX-512). The generator uses them as written; cost is accepted for v1 and measured. If it matters, the standard sign-flip trick (`Xor` with `1<<63`, signed compare) is applied inside the kernel.

## 12. Go codegen concerns and how each is handled

| concern | handling | verification |
|---|---|---|
| Bounds checks on every vector load/store | Three-index slices and `*[N]T` conversions, one check per block | `-gcflags=-d=ssa/check_bce` output in the hot loop |
| No compiler loop unrolling | Manual 4x unroll in `partition`; network fully unrolled by construction | code review |
| Inliner budget | `partitionVec`, `storeLR`, `load4`, `medianOf3`, `coex` are small leaf functions; nothing else is called from the partition loop | `-gcflags=-m=2` shows each as inlined |
| Register pressure in the network | Ordered so the compiler sees the paper's live ranges; measure spills | `go build -gcflags=-S`, count stack `MOVUPS`/`VMOVDQU` in the base case |
| Mask negation | `FromBits(^bits)` on AVX-512 (`KMOV` round trip); on AVX2 recompute the compare, never `FromBits` (vector mask expansion is emulated) | review |
| Emulated ops hiding cost | Every archsimd method used is checked against its `CPU Feature:` / `Emulated` annotation; the list is recorded in `gen/ops.md` | CI script greps the annotations for the used method set |
| AVX to SSE transition penalty in callers | `archsimd.ClearAVXUpperBits()` on every `sort*` return; check whether the compiler already emits `VZEROUPPER` and drop the call if so | objdump |
| AVX-512 frequency licensing | Provide the 256-bit kernel as a selectable target on AVX-512 machines (`SetTarget(AVX2)`); benchmark both | section 14 |
| Preemption and GC | Intrinsics are ordinary Go, async preemption works, vectors hold no pointers | no action |

## 13. Testing

Oracle: `slices.Sort` on a copy. Every test asserts the output is sorted and is a permutation of the input (sum and xor of bit patterns as a cheap check, full multiset comparison for `n <= 4096`).

Inputs, per key type:

- uniform random full-width; low entropy (16-bit values in 64-bit keys, the paper's degenerate-partition trigger); all equal; sorted; reverse sorted; sawtooth with periods 2, 3, 17, 256; organ pipe; `k` distinct values for `k ∈ {2, 3, 16, 256}`; alternating min/max.
- sizes: every `n` in `[0, 2048]` with three random seeds each (covers every base-case width and every remainder residue for `U = 32` and `U = 64`); then `n ∈ {4093, 65536, 1<<20, 1<<24}`.
- floats: mixes with NaN, `±Inf`, `±0`, subnormals; `float64` NaN payloads preserved (bit pattern check).
- adversarial: a fixed RNG seed with an input built by McIlroy's antiqsort against a scalar model of the pivot rule, to exercise the depth guard and heapsort path; assert the guard fired via a test hook.
- `go test -fuzz` on `[]int64` and `[]float32` with the permutation property.
- All of the above under `SetTarget(Scalar)`, `SetTarget(AVX2)`, `SetTarget(AVX512)` on an AVX-512 CI runner, plus one run on an AVX2-only runner (no SIGILL, correct results).

## 14. Benchmarking and acceptance

Harness: `go test -bench`, `benchstat`, single core, turbo disabled where the machine allows. Report MB/s of key bytes sorted.

Sizes: 1K, 64K (L2 resident), 1M, 100M keys. Types: i32, i64, f64, u64. Baselines: `slices.Sort`, and the paper's numbers as the ceiling (Skylake AVX-512, i64, 1M: 1137 MB/s vqsort, 118 MB/s libc++ `std::sort`).

Separately benchmark `partition` alone on 2^24 keys, the paper reports 11.6 GB/s for doubles on Skylake AVX-512. This isolates compiler codegen quality from algorithmic effects and is the first number to look at in M1.

Acceptance for v1: `>= 4x slices.Sort` on 1M i64 with AVX-512, `>= 2.5x` with AVX2. If M1 partition throughput is below 40 percent of the paper's figure after bounds-check and spill fixes, the fallback plan is a Go assembly `partition` (the loop is small and self-contained) while everything else stays in archsimd. That decision is made at the end of M1, not later.

## 15. Risks

1. archsimd API churn. Mitigation: all archsimd calls live in generated kernels; regenerating against a new API is a template edit.
2. Compiler leaves performance on the table (spills, unfused masked stores, redundant `KMOV`). Mitigation: assembly partition fallback per section 14; report findings upstream on golang/go#73787.
3. The 16-lane 32-bit AVX-512 kernel may not beat the AVX2 8-lane kernel because of the compress-expand cost. Mitigation: it is a per-type dispatch decision, made by benchmark.
4. AVX-512 frequency reduction on older Intel parts can make the 512-bit kernel slower than 256-bit at the system level. Mitigation: 256-bit target selectable at runtime; default chosen per benchmark, possibly per CPU family.
5. Memory bandwidth bound above L2 (paper, section 4.1). Not a risk to correctness; sets the ceiling for 100M-key numbers and for any later parallel version.

## 16. Milestones

- M0: package skeleton, generic API, scalar path (`slices.Sort`), test harness with all input families and the permutation oracle, `SetTarget`.
- M1: `int64` AVX-512 kernel with table-permute partition, scalar insertion-sort base case at `n <= 32`, median-of-3-of-3 scalar pivot. Partition microbenchmark. Go/no-go on codegen quality.
- M2: 16-row sorting network at full width, `c = N`; NBaseCase raised to `16·N`.
- M3: chunk sampling pivot with SFC64, degenerate-partition handling, depth guard and heapsort. Adversarial tests.
- M4: AVX2 `int64` kernel (table-permute-4), then AVX2 `int32`. Both targets under the forced-target test matrix.
- M5: kernel generator; all six key types on both targets; intermediate base-case widths; NaN prepass; unsigned paths.
- M6: benchmark matrix, per-type default target selection, `ClearAVXUpperBits` decision, docs.
- Later: descending order (second template instantiation), arm64 NEON via `Uint8x16.Permute` (`TBL`) as the compress, 128-bit keys via the paper's Algorithm 2, parallel driver (fork the recursion onto goroutines above a size threshold; separate memo).

## 17. Open questions

1. Does the compiler fuse `Compress(m).StoreArrayMasked(p, k)` into the memory-destination `VPCOMPRESS`? Relevant only to the remainder loop and to a future exact-store right side; not needed for v1.
2. `Int64x4.Permute` on AVX2: emulated via `VPERMD`, or gated on AVX512VL? Determines whether table-permute-4 can use it directly instead of the reshape path.
3. Base case width selection: does the paper's `c` by `n` rule pay off in Go once the network spills, or is a single full-width network plus insertion sort for small `n` within noise? M5 measurement.
4. Whether to expose the partition primitive (`vqsort.Partition(s, pivot) int`) as public API; useful for selection algorithms and for a future parallel driver.
5. 16-bit keys need `VPCOMPRESSW` (VBMI2) on AVX-512 and have no masked store on AVX2; the buffer-based store path from section 7.3 would have to become the main path. Deferred until a consumer exists.

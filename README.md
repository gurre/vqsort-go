# vqsort-go

A vectorized quicksort for Go, as a drop-in replacement for `slices.Sort`.

```go
slices.Sort(keys)  →  vqsort.Sort(keys)
```

Same signature, same result for every input, three to six times the throughput on
large numeric slices. Pure Go on `simd/archsimd` — no assembly, no cgo.

The algorithm is the vectorized quicksort of Blacher, Giesen, Wassenberg and
Sanders ([arXiv:2205.05982](https://arxiv.org/abs/2205.05982)): a bidirectional
in-place partition driven by SIMD compare-and-compress, a branch-free sorting
network for small ranges, and a heapsort fallback that keeps the worst case at
O(n log n).

## Results

Apple M4 Pro, single core, uniform random keys, against `slices.Sort`:

| keys | int32 | int64 | float32 | float64 |
|------|------:|------:|--------:|--------:|
| 64K  | 5.5×  | 2.7×  | 6.1×    | 3.2×    |
| 1M   | 5.9×  | 2.9×  | 6.6×    | 3.4×    |

At 1M int32 that is 410 MB/s of keys against 58 MB/s. Two pieces carry it: the
partition loop runs at 8.8 GB/s, 5.6× a branchless scalar partition over the
same data, and the base case is a branch-free sorting network that sorts 64 keys
in 42 ns — 30× the insertion sort it replaced.

Short slices go to `slices.Sort`, which is faster there, so you never pay to use
this package. The threshold is 64 keys for the 32-bit kernels and 8192 for the
64-bit ones, which move half as many keys per instruction and take longer to
amortize a partition call.

Reproduce with `GOEXPERIMENT=simd go test -bench BenchmarkSort`. Benchmarks
report MB/s of key bytes, `ns/key`, and `x_stdlib` measured in the same run.

## Requirements

Go 1.27, built with the SIMD experiment enabled:

```
GOEXPERIMENT=simd go build ./...
```

**That setting applies to the whole binary, not just this package** — your
builds, your CI, your Dockerfiles. Without it every call is still correct and
simply forwards to `slices.Sort`, so nothing breaks; you just get no speedup.
Call `vqsort.Accelerated[T]()` to find out which one you got.

Kernels ship for arm64 NEON. AVX2 and AVX-512 are not implemented yet; on amd64
the package currently forwards to `slices.Sort`.

## API

```go
func Sort[S ~[]E, E cmp.Ordered](x S)
func Accelerated[E cmp.Ordered]() bool

type Target int32
func SetTarget(t Target) (previous Target, err error)
func CurrentTarget() Target
```

`Sort` takes the same type set as `slices.Sort`. Element types with a vector
kernel — `int32`, `int64`, `int`, `float32`, `float64`, and any named type whose
underlying type is one of those — are sorted by it. Everything else `cmp.Ordered`
allows, including `string` and the unsigned and narrow integer types, forwards to
`slices.Sort`. So the rename always compiles and always produces the same answer.

Ordering matches `slices.Sort` exactly: NaNs sort before all other values,
`-0.0` and `+0.0` compare equal with unspecified relative order, and the sort is
not stable. It performs no heap allocation.

`SetTarget` forces a kernel for tests, benchmarks and diagnostics. It is
process-global, so call it from `main` or a test, never from a library. The Go
runtime's own switch also works and needs no code change:

```
GODEBUG=cpu.all=off   # force the scalar path
```

## Testing

```
go test ./...                     # exhaustive: every length to 2048, three seeds
go test -short ./...              # the boundaries that break things, seconds
go test -tags vqsortchecked ./... # same, with bounds checks back on
./mutate.sh                       # check the suite catches the bugs it should
mutest -packages .                # the same question asked of every defect site
```

The kernels index through unchecked pointer arithmetic, because the bounds check
sits on the innermost load and store of the partition loop and costs about nine
percent. `-tags vqsortchecked` swaps in a checked version of the same helper, so
an invariant violation panics instead of corrupting memory. Both configurations
pass, and CI runs both.

Every test asserts the output is sorted and is a permutation of the input, by
exact multiset comparison rather than a checksum — the input families are
deliberately low-entropy, which is where a checksum cancels. Float tests compare
raw bit patterns, so a lost NaN payload or a swapped signed zero is visible.

`mutate.sh` is the check on the checks: it injects thirteen real bugs this kernel
can have — a corrupted permutation table, an out-of-range table index that
`LookupOrZero` would silently turn into a zero, an off-by-one in the remainder
spill, the overlapping-store bug in the partition drain, an inverted network
comparator, a disabled depth guard — and fails if any of them survives the suite.
It has already paid for itself once, by showing that removing the depth guard
entirely left every test passing.

Eleven of those thirteen mutations live in files behind `goexperiment.simd &&
arm64`, so the script only means anything on arm64 — on amd64 it edits source the
compiler discards and reports the lot as survivors. CI runs it on the arm64 leg of
the matrix only.

[mutest](https://github.com/gurre/mutest) asks the same question of every defect
site rather than thirteen chosen ones, and tells an unreached site apart from a
survivor, which is the distinction `mutate.sh` cannot draw. A full sweep is a few
thousand defects and runs for hours against this suite, so it is a deliberate pass
rather than a CI step: `mutest -packages . -jobs 4`.

The sorting networks get their own tests: they are branch-free generated code, so
a miswired comparator sorts most inputs correctly. Those tests lean on the 0-1
principle — a data-oblivious network sorts every input exactly when it sorts
every 0/1 input — and on heavy-tie and single-element-walk cases.

## Status

Not released. arm64 only; amd64 kernels and unsigned key types are
unimplemented. The `archsimd` API is experimental and is expected to change
again in Go 1.28.

`go generate ./...` regenerates the float kernels and the sorting networks; the
generated files are checked in and CI verifies they are up to date.

## License

MIT — see [LICENSE](LICENSE).

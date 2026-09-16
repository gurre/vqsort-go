package vqsort

import (
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"
)

// A Target is a sorting kernel: either the portable scalar path or a specific
// instruction set extension.
//
// The instruction-set constants are declared per architecture, so NEON exists
// only in arm64 builds and AVX2 and AVX512 only in amd64 builds. Naming the
// wrong one for the target architecture is a compile error rather than a call
// that always fails.
type Target int32

// Kernel identities. The exported names live in target_arm64.go and
// target_amd64.go; these values exist on every architecture so that String and
// kernel dispatch compile everywhere.
const (
	// Auto selects the fastest kernel available on this CPU. It is the
	// default, and it is never reported by CurrentTarget, which always
	// reports the kernel Auto resolved to.
	Auto Target = iota

	// Scalar forwards to slices.Sort.
	Scalar

	neon
	avx2
	avx512
)

func (t Target) String() string {
	switch t {
	case Auto:
		return "auto"
	case Scalar:
		return "scalar"
	case neon:
		return "neon"
	case avx2:
		return "avx2"
	case avx512:
		return "avx512"
	}
	return fmt.Sprintf("Target(%d)", int32(t))
}

// current holds the resolved kernel, never Auto. Sort reads it once per call,
// so SetTarget racing with a sort is not a data race.
var current atomic.Int32

func init() { current.Store(int32(bestTarget())) }

// ErrUnavailableTarget is returned by SetTarget when the requested kernel is
// not compiled into this binary or not supported by this CPU.
var ErrUnavailableTarget = errors.New("vqsort: target unavailable")

// SetTarget forces Sort to use kernel t and returns the kernel that was
// previously in use. Passing Auto restores the fastest kernel this CPU
// supports.
//
// If t is not available - because the binary was built without
// GOEXPERIMENT=simd, because it names another architecture's instruction set,
// or because this CPU lacks the feature - SetTarget changes nothing and returns
// an error wrapping ErrUnavailableTarget. It does not panic, so a target read
// from configuration cannot crash a process on a machine that happens to lack
// the extension.
//
// SetTarget is process-global. Call it from main or from a test, never from a
// reusable library: forcing a slower kernel silently changes the performance,
// though never the results, of every other caller in the process.
func SetTarget(t Target) (previous Target, err error) {
	previous = CurrentTarget()
	resolved := t
	if t == Auto {
		resolved = bestTarget()
	}
	if !targetAvailable(resolved) {
		return previous, fmt.Errorf("%w: %s on %s/%s", ErrUnavailableTarget, t, runtime.GOOS, runtime.GOARCH)
	}
	current.Store(int32(resolved))
	return previous, nil
}

// CurrentTarget reports the kernel Sort is using. It never returns Auto: a
// process that has not called SetTarget reports whatever Auto resolved to at
// startup.
func CurrentTarget() Target { return Target(current.Load()) }

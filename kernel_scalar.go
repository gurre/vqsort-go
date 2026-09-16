//go:build !(goexperiment.simd && arm64)

package vqsort

import "slices"

// This file supplies the kernel entry points for every build that has no vector
// kernel: any binary built without GOEXPERIMENT=simd, and any architecture
// vqsort does not implement yet. Results are identical to the vector kernels;
// only throughput differs.

// These exist on every build so that tests can refer to them; with no kernels
// compiled in they have no effect.
var (
	vectorFloor32 = 64
	vectorFloor64 = 8192
)

func bestTarget() Target { return Scalar }

func targetAvailable(t Target) bool { return t == Scalar }

func sortInt32(s []int32)     { slices.Sort(s) }
func sortInt64(s []int64)     { slices.Sort(s) }
func sortFloat32(s []float32) { slices.Sort(s) }
func sortFloat64(s []float64) { slices.Sort(s) }

// kernelReady reports whether element kind k has a vector kernel under target
// t. In a build with no kernels, nothing is accelerated.
func kernelReady(elemKind, Target) bool { return false }

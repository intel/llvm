// REQUIRES: (gpu && (hip || cuda)), cpu
// RUN: %{build} %{embed-ir} -O2 -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --implicit-check-not "WRONG a VALUE" --implicit-check-not "WRONG b VALUE"

// Test caching for JIT fused kernels when devices with different architectures
// are involved.

#include "./jit_caching_multitarget_common.h"

// Initial invocation
// CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

// Identical invocation, should lead to JIT cache hit.
// CHECK-NEXT: JIT DEBUG: Re-using cached JIT kernel
// CHECK-NEXT: INFO: Re-using existing device binary for fused kernel

// Invocation with a device with a different architecture. Should trigger JIT
// compilation.
// CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

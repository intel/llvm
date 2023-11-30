// REQUIRES: fusion,gpu,opencl
// RUN: %{build} -O2 -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 env ONEAPI_DEVICE_SELECTOR="opencl:cpu,gpu" %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --implicit-check-not "COMPUTATION ERROR" --implicit-check-not "WRONG INTERNALIZATION"

#include "./jit_caching.h"

// Test caching for JIT fused kernels. Also test for debug messages being
// printed when SYCL_RT_WARNING_LEVEL=1.
//
// Test when combining SPIR-V devices.

// Initial invocation
// CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

// Identical invocation, should lead to JIT cache hit.
// CHECK: JIT DEBUG: Re-using cached JIT kernel
// CHECK: INFO: Re-using existing device binary for fused kernel

// Invocation with a different beta. Because beta was identical to alpha so far,
// this should lead to a cache miss.
// CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

// Invocation with barrier insertion should lead to a cache miss.
// CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

// Invocation with different internalization target should lead to a cache miss.
// CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

// Invocation with a different gamma should lead to a cache miss because gamma
// participates in constant propagation.
// CHECK: JIT DEBUG: Compiling new kernel, no suitable cached kernel found

// Invocation on a different device. Both are SPIR-V, so cache should be hit.
// CHECK: INFO: Re-using existing device binary for fused kernel

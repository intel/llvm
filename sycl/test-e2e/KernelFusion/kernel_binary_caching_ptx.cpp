// REQUIRES: fusion,cuda,gpu
// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu,cuda:gpu %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --implicit-check-not "COMPUTATION ERROR"

// Test caching for kernel binary images. Also test for debug messages being
// printed when SYCL_RT_WARNING_LEVEL=1.
//
// Test when combining SPIR-V and PTX devices.

#include "./kernel_binary_caching.h"

// Initial invocation
// CHECK: INFO: No cached kernel binary for '_ZTS7Kernel0'. Creating a new one.
// CHECK: INFO: No cached kernel binary for '_ZTS7Kernel1'. Creating a new one.
// CHECK: INFO: No cached kernel binary for '_ZTS7Kernel2'. Creating a new one.

// Identical invocation, should lead to kernel binary cache hit.
// CHECK: INFO: Re-using cached kernel binary for '_ZTS7Kernel0'
// CHECK: INFO: Re-using cached kernel binary for '_ZTS7Kernel1'
// CHECK: INFO: Re-using cached kernel binary for '_ZTS7Kernel2'

// Initial invocation for CUDA device
// CHECK: INFO: No cached kernel binary for '_ZTS7Kernel0'. Creating a new one.
// CHECK: INFO: No cached kernel binary for '_ZTS7Kernel1'. Creating a new one.
// CHECK: INFO: No cached kernel binary for '_ZTS7Kernel2'. Creating a new one.

// Identical invocation, should lead to kernel binary cache hit.
// CHECK: INFO: Re-using cached kernel binary for '_ZTS7Kernel0'
// CHECK: INFO: Re-using cached kernel binary for '_ZTS7Kernel1'
// CHECK: INFO: Re-using cached kernel binary for '_ZTS7Kernel2'

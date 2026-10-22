// REQUIRES: arch-intel_gpu_cri
// UNSUPPORTED: any-target-is-nvidia, any-target-is-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} -DFULL_ATOMIC16_COVERAGE -DFULL_ATOMIC32_COVERAGE -DFULL_ATOMIC64_COVERAGE -fsycl-device-code-split=per_kernel -Xspirv-translator=spir64 -spirv-ext=+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics -o %t.cri.out
// RUN: %{run} %t.cri.out

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define TEST_GENERIC_IN_LOCAL 1

#include "add.h"

int main() { add_test_all<access::address_space::generic_space>(); }

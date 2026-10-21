// REQUIRES: arch-intel_gpu_cri
// RUN: %{build} -DFULL_ATOMIC16_COVERAGE -DFULL_ATOMIC32_COVERAGE -DFULL_ATOMIC64_COVERAGE -fsycl-device-code-split=per_kernel -Xspirv-translator=spir64 -spirv-ext=+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics -o %t.cri.out
// RUN: %{run} %t.cri.out

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15791

#include "exchange.h"

int main() { exchange_test_all<access::address_space::global_space>(); }

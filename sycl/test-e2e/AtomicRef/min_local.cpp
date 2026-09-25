// RUN: %if arch-intel_gpu_cri %{ %{build} -DFULL_ATOMIC16_COVERAGE -DFULL_ATOMIC32_COVERAGE -DFULL_ATOMIC64_COVERAGE -fsycl-device-code-split=per_kernel -Xspirv-translator -spirv-ext=+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics -o %t.out %}
// RUN: %if arch-intel_gpu_cri %{ %{run} %t.out %}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "min.h"

int main() { min_test_all<access::address_space::local_space>(); }

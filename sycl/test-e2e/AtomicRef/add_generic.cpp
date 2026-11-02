// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} -o %t.out
// RUN: %{build} %{full_atomic_flags} -o %t.cri.out
// RUN: %if arch-intel_gpu_cri %{ %{run} %t.cri.out %} %else %{ %{run} %t.out %}

#include "add.h"

int main() { add_test_all<access::address_space::generic_space>(); }

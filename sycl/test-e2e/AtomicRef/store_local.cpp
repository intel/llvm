// UNSUPPORTED: any-target-is-nvidia, any-target-is-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} -o %t.out
// RUN: %{build} %{full_atomic_flags} -o %t.cri.out
// RUN: %if arch-intel_gpu_cri %{ %{run} %t.cri.out %} %else %{ %{run} %t.out %}

#include "store.h"

int main() { store_test_all<access::address_space::local_space>(); }

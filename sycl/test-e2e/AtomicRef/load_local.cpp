// UNSUPPORTED: any-target-is-nvidia, any-target-is-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build_atomic} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: windows && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/23415

#include "load.h"

int main() { load_test_all<access::address_space::local_space>(); }

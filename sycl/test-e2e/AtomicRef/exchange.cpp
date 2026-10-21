// UNSUPPORTED: any-target-is-nvidia, any-target-is-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build_atomic} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15791

#include "exchange.h"

int main() { exchange_test_all<access::address_space::global_space>(); }

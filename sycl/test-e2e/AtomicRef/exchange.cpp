// RUN: %{build} -o %t.out
// RUN: %{build_full_atomic} -o %t.full.out
// RUN: %if target-spir && !spirv-backend && aspect-ext_oneapi_atomic16 && aspect-fp16 && aspect-atomic64 && aspect-fp64 %{ %{run} %t.full.out %} %else %{ %{run} %t.out %}

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15791

#include "exchange.h"

int main() { exchange_test_all<access::address_space::global_space>(); }

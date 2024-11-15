// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip_amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15791

#include "exchange.h"

int main() { exchange_test_all<access::address_space::global_space>(); }

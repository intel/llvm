// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// https://github.com/intel/llvm/issues/15791
// UNSUPPORTED: hip_amd

#include "exchange.h"

int main() { exchange_test_all<access::address_space::global_space>(); }

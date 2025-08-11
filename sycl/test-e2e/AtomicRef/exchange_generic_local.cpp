// XFAIL: hip
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19077
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define TEST_GENERIC_IN_LOCAL 1

#include "exchange.h"

int main() { exchange_test_all<access::address_space::generic_space>(); }

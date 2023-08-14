// REQUIRES: hip_amd
// RUN: %{build} -mllvm -debug-only="regalloc" -o %t.out
// RUN: env AMD_LOG_LEVEL=3 %{run} %t.out

#define TEST_GENERIC_IN_LOCAL 1

#include "store.h"

int main() { store_test_all<access::address_space::generic_space>(); }

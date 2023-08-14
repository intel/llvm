// REQUIRES: hip_amd
// RUN: %{build} -mllvm -debug-only="regalloc" -o %t.out
// RUN: env AMD_LOG_LEVEL=3 %{run} %t.out

#include "or.h"

int main() { or_test_all<access::address_space::global_space>(); }

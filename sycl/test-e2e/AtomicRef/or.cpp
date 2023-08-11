// REQUIRES: hip_amd
// RUN: %{build} -mllvm -debug-only="regalloc,si-insert-waitcnts,si-fix-sgpr-copies,si-fix-vgpr-copies,si-instr-info" -o %t.out
// RUN: AMD_LOG_LEVEL=3 %{run} %t.out

#include "or.h"

int main() { or_test_all<access::address_space::global_space>(); }

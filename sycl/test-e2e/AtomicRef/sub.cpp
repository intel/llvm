// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/23748

#define TEST_ATOMIC_FLOAT
#include "sub.h"

int main() { sub_test_all<access::address_space::global_space>(); }

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/23748

#define TEST_ATOMIC_FLOAT
#include "add.h"

int main() { add_test_all<access::address_space::generic_space>(); }

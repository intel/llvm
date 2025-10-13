// UNSUPPORTED: target-nvidia,cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20109

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "max.h"

int main() { max_test_all<access::address_space::global_space>(); }

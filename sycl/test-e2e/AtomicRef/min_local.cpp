// UNSUPPORTED: target-nvidia,cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20109

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "min.h"

int main() { min_test_all<access::address_space::local_space>(); }

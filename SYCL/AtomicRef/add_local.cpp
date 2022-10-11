// See https://github.com/intel/llvm-test-suite/issues/867 for detailed status
// UNSUPPORTED: hip

// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "add.h"

int main() { add_test_all<access::address_space::local_space>(); }

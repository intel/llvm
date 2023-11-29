// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define SYCL_USE_NATIVE_FP_ATOMICS
#define FP_TESTS_ONLY

#include "max.h"

int main() { max_test_all<access::address_space::generic_space>(); }

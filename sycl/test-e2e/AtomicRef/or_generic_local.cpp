// RUN: %{build} -o %t.out
// RUN: %{build_full_atomic} -o %t.full.out
// RUN: %if target-spir && !spirv-backend && aspect-ext_oneapi_atomic16 && aspect-atomic64 %{ %{run} %t.full.out %} %else %{ %{run} %t.out %}

#define TEST_GENERIC_IN_LOCAL 1

#include "or.h"

int main() { or_test_all<access::address_space::generic_space>(); }

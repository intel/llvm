// RUN: %{build} -o %t.out
// RUN: %{build_full_atomic} -o %t.full.out
// RUN: %if target-spir && !spirv-backend && aspect-ext_oneapi_atomic16 && aspect-atomic64 %{ %{run} %t.full.out %} %else %{ %{run} %t.out %}

#include "xor.h"

int main() { xor_test_all<access::address_space::local_space>(); }

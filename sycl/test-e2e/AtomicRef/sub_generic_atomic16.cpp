// REQUIRES: target-spir
// UNSUPPORTED: spirv-backend
// UNSUPPORTED-INTENDED: SPIR-V backend invocation lacks atomic16/bfloat16
// extensions
// RUN: %{build} %{full_atomic16_flags} -o %t.out
// RUN: %if aspect-ext_oneapi_atomic16 && aspect-fp16 %{ %{run} %t.out %} %else %{ echo "Skipped: requires atomic16 and fp16" %}

#include "sub.h"

int main() { sub_test_all<access::address_space::generic_space>(); }

// RUN: %{build} %{full_atomic64_flags} -o %t.out
// RUN: %if aspect-atomic64 && aspect-fp64 %{ %{run} %t.out %} %else %{ echo "Skipped: requires atomic64 and fp64" %}

#include "sub.h"

int main() { sub_test_all<access::address_space::generic_space>(); }

// RUN: %{build} %{full_atomic32_flags} -o %t.out
// RUN: %{run} %t.out

#include "add.h"

int main() { add_test_all<access::address_space::generic_space>(); }

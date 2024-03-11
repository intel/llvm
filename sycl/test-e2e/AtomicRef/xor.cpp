// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "xor.h"

int main() { xor_test_all<access::address_space::global_space>(); }

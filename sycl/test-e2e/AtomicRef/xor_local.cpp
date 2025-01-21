// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "xor.h"

int main() { xor_test_all<access::address_space::local_space>(); }

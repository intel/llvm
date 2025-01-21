// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "or.h"

int main() { or_test_all<access::address_space::generic_space>(); }

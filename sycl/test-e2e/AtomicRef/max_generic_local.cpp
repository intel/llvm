// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define TEST_GENERIC_IN_LOCAL 1

#include "max.h"

int main() { max_test_all<access::address_space::generic_space>(); }

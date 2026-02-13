// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "min.h"

int main() { min_test_all<access::address_space::global_space>(); }

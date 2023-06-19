// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "load.h"

int main() { load_test_all<access::address_space::generic_space>(); }

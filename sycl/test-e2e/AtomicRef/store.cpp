// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "store.h"

int main() { store_test_all<access::address_space::global_space>(); }

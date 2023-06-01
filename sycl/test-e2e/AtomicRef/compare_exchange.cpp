// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "compare_exchange.h"

int main() { compare_exchange_test_all<access::address_space::global_space>(); }

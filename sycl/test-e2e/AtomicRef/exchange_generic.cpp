// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "exchange.h"

int main() { exchange_test_all<access::address_space::generic_space>(); }

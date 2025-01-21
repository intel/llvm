// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "max.h"

int main() { max_test_all<access::address_space::local_space>(); }

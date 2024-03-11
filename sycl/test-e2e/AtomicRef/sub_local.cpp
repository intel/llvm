// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "sub.h"

int main() { sub_test_all<access::address_space::local_space>(); }

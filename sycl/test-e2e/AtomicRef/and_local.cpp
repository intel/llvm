// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "and.h"

int main() { and_test_all<access::address_space::local_space>(); }

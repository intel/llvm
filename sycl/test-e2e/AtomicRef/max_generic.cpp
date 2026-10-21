// UNSUPPORTED: any-target-is-nvidia, any-target-is-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build_atomic} -o %t.out
// RUN: %{run} %t.out

#include "max.h"

int main() { max_test_all<access::address_space::generic_space>(); }

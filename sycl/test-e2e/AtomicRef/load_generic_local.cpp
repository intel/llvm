// UNSUPPORTED: any-target-is-nvidia, any-target-is-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build_atomic} -o %t.out
// RUN: %{run} %t.out

#define TEST_GENERIC_IN_LOCAL 1

#include "load.h"

int main() { load_test_all<access::address_space::generic_space>(); }

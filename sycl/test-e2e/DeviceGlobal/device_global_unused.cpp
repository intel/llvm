// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that device_global with no kernel uses can be copied to and from.

#include "device_global_unused.hpp"

int main() { return test(); }

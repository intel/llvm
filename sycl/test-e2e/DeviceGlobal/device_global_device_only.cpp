// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287
//
// Tests basic device_global access through device kernels.

#include "device_global_device_only.hpp"

int main() { return test(); }

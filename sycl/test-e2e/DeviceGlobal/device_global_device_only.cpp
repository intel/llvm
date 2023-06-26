// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The HIP and OpenCL GPU backends do not currently support device_global
// backend calls.
// UNSUPPORTED: hip || (opencl && gpu)
//
// Tests basic device_global access through device kernels.

#include "device_global_device_only.hpp"

int main() { return test(); }

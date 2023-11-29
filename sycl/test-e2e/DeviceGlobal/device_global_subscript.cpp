// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The HIP and OpenCL GPU backends do not currently support device_global
// backend calls.
// UNSUPPORTED: hip || (opencl && gpu)
//
// Tests operator[] on device_global.

#include "device_global_subscript.hpp"

int main() { return test(); }

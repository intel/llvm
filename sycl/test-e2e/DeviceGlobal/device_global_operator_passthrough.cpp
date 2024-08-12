// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The OpenCL GPU backends do not currently support device_global backend
// calls.
// UNSUPPORTED: opencl && gpu
//
// Tests the passthrough of operators on device_global.

#include "device_global_operator_passthrough.hpp"

int main() { return test(); }

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287
//
// Tests the passthrough of operators on device_global.

#include "device_global_operator_passthrough.hpp"

int main() { return test(); }

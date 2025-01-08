// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287
//
// Tests operator[] on device_global.

#include "device_global_subscript.hpp"

int main() { return test(); }

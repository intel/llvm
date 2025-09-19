// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287
// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-INTENDED: Currently mark Native CPU as unsupported, it should be
// investigated and tracked post team transfer.
//
// Tests operator-> on device_global.

#include "device_global_arrow.hpp"

int main() { return test(); }

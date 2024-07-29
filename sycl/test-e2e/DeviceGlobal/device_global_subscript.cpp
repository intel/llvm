// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The OpenCL GPU backends do not currently support device_global backend
// calls.
// TODO: Reenable linux/windows, see https://github.com/intel/llvm/issues/14598
// UNSUPPORTED: opencl && gpu, linux, windows
//
// Tests operator[] on device_global.

#include "device_global_subscript.hpp"

int main() { return test(); }

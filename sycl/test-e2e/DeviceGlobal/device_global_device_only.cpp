// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The OpenCL GPU backends do not currently support device_global backend
// calls.
// TODO: Reenable linux/windows, see https://github.com/intel/llvm/issues/14598
// UNSUPPORTED: opencl && gpu, linux, windows
//
// Tests basic device_global access through device kernels.

#include "device_global_device_only.hpp"

int main() { return test(); }

// RUN: %{build} -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE -o %t.out
// RUN: %{run} %t.out
//
// The HIP and OpenCL GPU backends do not currently support device_global
// backend calls.
// UNSUPPORTED: hip || (opencl && gpu)
//
// Tests basic device_global with device_image_scope access through device
// kernels.
// NOTE: USE_DEVICE_IMAGE_SCOPE needs both kernels to be in the same image so
//       we set -fsycl-device-code-split=per_source.

#include "device_global_device_only.hpp"

int main() { return test(); }

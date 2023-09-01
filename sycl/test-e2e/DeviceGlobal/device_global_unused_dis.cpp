// RUN: %{build} -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE -o %t.out
// RUN: %{run} %t.out

// Tests that device_global with device_image_scope and no kernel uses can be
// copied to and from.
// NOTE: USE_DEVICE_IMAGE_SCOPE needs both kernels to be in the same image so
//       we set -fsycl-device-code-split=per_source.

#include "device_global_unused.hpp"

int main() { return test(); }

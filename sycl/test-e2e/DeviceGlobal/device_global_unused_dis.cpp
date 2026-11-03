// RUN: %{build} -DUSE_DEVICE_IMAGE_SCOPE -o %t.out
// RUN: %{run} %t.out

// Tests that copying to or from a device_global with device_image_scope that
// no kernel uses throws errc::kernel_not_supported.

#include "common.hpp"

device_global<int, TestProperties> memcpyDeviceGlobal;
device_global<int, TestProperties> copyDeviceGlobal;

namespace {
template <typename FnT> bool throwsKernelNotSupported(FnT kFn) {
  try {
    kFn();
  } catch (const sycl::exception &e) {
    return e.code() == errc::kernel_not_supported;
  }
  return false;
}
} // namespace

int main() {
  queue q;
  int value = 42;
  int failures = 0;

  failures +=
      !throwsKernelNotSupported([&] { q.memcpy(memcpyDeviceGlobal, &value); });
  failures +=
      !throwsKernelNotSupported([&] { q.memcpy(&value, memcpyDeviceGlobal); });
  failures +=
      !throwsKernelNotSupported([&] { q.copy(&value, copyDeviceGlobal); });
  failures +=
      !throwsKernelNotSupported([&] { q.copy(copyDeviceGlobal, &value); });

  return failures;
}

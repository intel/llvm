// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify=checks-on -Xclang -verify-ignore-unexpected=warning,note %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -DSYCL_DISABLE_DEVICE_COPYABLE_CHECKS -Xclang -verify=checks-off -Xclang -verify-ignore-unexpected=warning,note %s

// checks-off-no-diagnostics

#include <sycl/detail/core.hpp>

// A user-provided destructor is enough to make this neither device copyable nor
// eligible for the deprecated trivially-copyable exception.
struct NotDeviceCopyable {
  ~NotDeviceCopyable() {}
};

int main() {
  NotDeviceCopyable Val;
  // checks-on-error@*:* {{The specified type is not device copyable}}
#ifdef SYCL_DISABLE_DEVICE_COPYABLE_CHECKS
  static_assert(sycl::is_device_copyable_v<NotDeviceCopyable>);
#else
  static_assert(!sycl::is_device_copyable_v<NotDeviceCopyable>);
#endif
  sycl::queue{}.single_task([=] { (void)Val; });
}

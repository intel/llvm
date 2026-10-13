// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify=checks-on -Xclang -verify-ignore-unexpected=warning,note %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -DSYCL_DISABLE_DEVICE_COPYABLE_CHECKS -Xclang -verify=checks-off -Xclang -verify-ignore-unexpected=warning,note %s

// checks-off-no-diagnostics

#include <sycl/detail/core.hpp>

// A user-provided destructor is enough to make this neither device copyable nor
// eligible for the deprecated trivially-copyable exception.
struct NotDeviceCopyable {
  ~NotDeviceCopyable() {}
};

// The macro only turns the checks off, it must not change what the user facing
// trait reports about a type.
static_assert(!sycl::is_device_copyable_v<NotDeviceCopyable>);

int main() {
  NotDeviceCopyable Val;
  // checks-on-error@*:* {{The specified type is not device copyable}}
  sycl::queue{}.single_task([=] { (void)Val; });

  // checks-on-error@*:* {{a buffer must be device copyable}}
  sycl::buffer<NotDeviceCopyable, 1> Buf{sycl::range<1>{1}};
}

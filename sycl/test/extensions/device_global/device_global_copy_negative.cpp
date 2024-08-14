// RUN: %clangxx -std=c++23 -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Tests that the copy ctor on device_global with device_image_scope is
// unavailable.

#include <sycl/sycl.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

using device_image_properties =
    decltype(oneapiext::properties{oneapiext::device_image_scope});

oneapiext::device_global<const int, device_image_properties> DGInit{3};
oneapiext::device_global<const int, device_image_properties> DGCopy{DGInit};

// expected-error@sycl/ext/oneapi/device_global/device_global.hpp:* {{call to deleted constructor}}

int main() { return 0; }

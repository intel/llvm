// RUN: %clangxx -std=c++23 -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Tests that the copy ctor on device_global with device_image_scope is
// unavailable.

#include <sycl/sycl.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

using device_image_properties =
    decltype(oneapiext::properties{oneapiext::device_image_scope});

// expected-error@sycl/ext/oneapi/device_global/device_global.hpp:* {{call to deleted constructor}}
oneapiext::device_global<const int, device_image_properties> DGInit1{3};
oneapiext::device_global<const int, device_image_properties> DGCopy1{DGInit1};

// expected-error@sycl/ext/oneapi/device_global/device_global.hpp:* {{call to deleted constructor}}
oneapiext::device_global<int, device_image_properties> DGInit2{3};
oneapiext::device_global<int, device_image_properties> DGCopy2{DGInit2};

// expected-error@+2 {{call to deleted constructor}}
oneapiext::device_global<int, device_image_properties> DGInit3{3};
oneapiext::device_global<float, device_image_properties> DGCopy3{DGInit3};

// expected-error@+2 {{call to deleted constructor}}
oneapiext::device_global<const int> DGInit4{3};
oneapiext::device_global<const int, device_image_properties> DGCopy4{DGInit4};

int main() { return 0; }

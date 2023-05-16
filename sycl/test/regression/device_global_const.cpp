// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

namespace experimental = sycl::ext::oneapi::experimental;

const experimental::device_global<int> DeviceGlobal;
const experimental::device_global<int, decltype(experimental::properties{
                                           experimental::device_image_scope})>
    ScopedDeviceGlobal;

// RUN: %clangxx -S -emit-llvm -fsycl-device-only %s -o - | FileCheck %s

// Checks that the host_access property doesn't get represented when there is no
// device_image_scope property.

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

static device_global<int,
                     decltype(properties(device_image_scope, host_access_read))>
    DeviceGlobalDeviceImageScoped;
static device_global<int, decltype(properties(host_access_read))>
    DeviceGlobalFullScoped;

SYCL_EXTERNAL void ignore_host_access() {
  DeviceGlobalFullScoped = 42;
  DeviceGlobalDeviceImageScoped = 42;
}

// CHECK-DAG: @_ZL29DeviceGlobalDeviceImageScoped = {{.*}} #[[DISAttrs:[0-9]+]]
// CHECK-DAG: @_ZL22DeviceGlobalFullScoped = {{.*}} #[[FSAttrs:[0-9]+]]
// CHECK-DAG: attributes #[[DISAttrs:[0-9]+]] = { {{.*}}"sycl-host-access"
// CHECK-NOT: attributes #[[FSAttrs:[0-9]+]] = { {{.*}}"sycl-host-access"

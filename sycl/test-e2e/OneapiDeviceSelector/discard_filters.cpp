// REQUIRES: gpu
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/trivial.cpp -o %t.out

// Test discard filters in ONEAPI_DEVICE_SELECTOR.
// RUN: env ONEAPI_DEVICE_SELECTOR="!*:gpu" %t.out | FileCheck %s --allow-empty
// CHECK-NOT: {{.*}}Device:{{.*}}{{gpu|GPU|Gpu}}{{.*}}

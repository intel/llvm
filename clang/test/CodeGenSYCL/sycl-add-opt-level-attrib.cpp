// RUN: %clangxx %s -O0 -S -o %t.ll -fsycl-device-only
// RUN: FileCheck %s --input-file %t.ll -check-prefixes=CHECK-IR
// CHECK-IR: define weak_odr dso_local spir_kernel void @{{.*}}main{{.*}}sycl{{.*}}handler{{.*}}() #[[ATTR:[0-9]+]]
// CHECK-IR: attributes #[[ATTR]] = { {{.*}} "sycl-optlevel"="0" {{.*}}}

// This test checks adding of the attribute 'sycl-optlevel'
// by the clang front-end

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {});
  });
  return 0;
}


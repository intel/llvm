// RUN: %clangxx %s -O0 -S -o %t.ll -fsycl-device-only
// RUN: FileCheck %s --input-file %t.ll -check-prefixes=CHECK-IR
// CHECK-IR: define weak_odr dso_local spir_kernel void @{{.*}}main{{.*}}sycl{{.*}}handler{{.*}}() #[[ATTR:[0-9]+]]
// CHECK-IR: attributes #[[ATTR]] = { {{.*}} "sycl-device-compile-optlevel"="0" {{.*}}}

// RUN: %clangxx %s -O0 -o %t.bc -fsycl-device-only
// RUN: sycl-post-link -split=source -symbols -S %t.bc -o %t.table
// RUN: FileCheck %s -input-file=%t.table
// RUN: FileCheck %s -input-file=%t_0.prop --check-prefixes CHECK-OPT-LEVEL-PROP

// CHECK: [Code|Properties|Symbols]
// CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym

// CHECK-OPT-LEVEL-PROP: OptLevel=1|0

// This test checks adding of the attribute 'sycl-device-compile-optlevel'
// by the clang front-end
// This test also checks parsing of the attribute 'sycl-device-compile-optlevel'
// by the sycl-post-link-tool:
// Splitting happens as usual.
// - sycl-post-link adds 'OptLevel' property to the device binary

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {});
  });
  return 0;
}


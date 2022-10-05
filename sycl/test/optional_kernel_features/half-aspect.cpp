// RUN: %clangxx %s -o %t.bc -fsycl-device-only
// RUN: llvm-dis %t.bc -o %t.ll
// RUN: FileCheck %s --input-file %t.ll

// CHECK: !sycl_types_that_use_aspects = !{![[#MDNUM:]]}
// CHECK: ![[#MDNUM]] = !{!"class.sycl::_V1::detail::half_impl::half", i32 5}
// CHECK: !{{.*}} = !{!"fp16", i32 5}

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      sycl::half h;
      h = 10.0;
    });
  });
  return 0;
}

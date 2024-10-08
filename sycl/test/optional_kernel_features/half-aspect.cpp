// RUN: %clangxx %s -S -o %t.ll -fsycl-device-only -Xclang -disable-llvm-passes
// RUN: FileCheck %s --input-file %t.ll

// CHECK: !sycl_types_that_use_aspects = !{![[#MDNUM:]]}
// CHECK: ![[#MDNUM]] = !{!"class.sycl::_V1::detail::half_impl::half", i32 [[#ASPECT_NUM:]]}
// CHECK: !{{.*}} = !{!"fp16", i32 [[#ASPECT_NUM]]}

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

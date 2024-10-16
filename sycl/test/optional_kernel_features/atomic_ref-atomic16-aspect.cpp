// RUN: %clangxx %s -S -o %t.ll -fsycl-device-only -Xclang -disable-llvm-passes
// RUN: FileCheck %s --input-file %t.ll

// CHECK: !sycl_types_that_use_aspects = !{![[#MDNUM1:]], ![[#MDNUM2:]]}
// CHECK: ![[#MDNUM1]] = !{!"class.sycl::_V1::detail::half_impl::half", i32 [[#FP16_ASPECT_NUM:]]}
// CHECK: ![[#MDNUM2]] = !{!"class.sycl::_V1::detail::atomic_ref_impl", i32 [[#ATOMIC16_ASPECT_NUM:]]}
// CHECK: !{{.*}} = !{!"fp16", i32 [[#FP16_ASPECT_NUM]]}
// CHECK: !{{.*}} = !{!"ext_oneapi_atomic16", i32 [[#ATOMIC16_ASPECT_NUM]]}

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      // TODO: also test short, unsigned short and bfloat16 when available
      sycl::half val_half = 1.0;
      auto ref_half =
          sycl::atomic_ref<sycl::half, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(val_half);
    });
  });
  return 0;
}

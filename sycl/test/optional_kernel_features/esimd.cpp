// RUN: %clangxx %s -S -o %t.ll -fsycl-device-only -Xclang -disable-llvm-passes
// RUN: FileCheck %s --input-file %t.ll

// CHECK: !sycl_types_that_use_aspects = !{![[#MDNUM1:]], ![[#MDNUM2:]]}
// CHECK: ![[#MDNUM1]] = !{!"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl", i32 [[#ASPECT_NUM:]]}
// CHECK: ![[#MDNUM2]] = !{!"class.sycl::_V1::ext::intel::esimd::detail::simd_view_impl", i32 [[#ASPECT_NUM]]}
// CHECK: !{{.*}} = !{!"ext_intel_esimd", i32 [[#ASPECT_NUM]]}

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) SYCL_ESIMD_KERNEL {
    h.single_task([=]() {
      simd<int, 16> va;
      auto view = va.select<1, 1>();
    });
  });
  return 0;
}

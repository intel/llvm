// RUN: %clangxx %s -S -o %t.ll -fsycl-device-only -Xclang -disable-llvm-passes
// RUN: FileCheck %s --input-file %t.ll

// CHECK: !sycl_types_that_use_aspects = !{![[#MDNUM1:]], ![[#MDNUM2:]], ![[#MDNUM3:]], ![[#MDNUM4:]]}
// CHECK: ![[#MDNUM1]] = !{!"class.sycl::_V1::detail::atomic_ref_impl.20", i32 [[#ASPECT_NUM:]]}
// CHECK-NEXT: ![[#MDNUM2]] = !{!"class.sycl::_V1::detail::atomic_ref_impl", i32 [[#ASPECT_NUM:]]}
// CHECK-NEXT: ![[#MDNUM3]] = !{!"class.sycl::_V1::detail::atomic_ref_impl.2", i32 [[#ASPECT_NUM:]]}
// CHECK-NEXT: ![[#MDNUM4]] = !{!"class.sycl::_V1::detail::atomic_ref_impl.7", i32 [[#ASPECT_NUM:]]}
// CHECK: !{{.*}} = !{!"atomic64", i32 [[#ASPECT_NUM]]}

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      double val_double = 100.0;
      // uses atomic64 aspect
      auto ref_double =
          sycl::atomic_ref<double, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(
              val_double);
      long long val_longlong = 101;
      // uses atomic64 aspect
      auto ref_longlong =
          sycl::atomic_ref<long long, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(
              val_longlong);
      uint64_t val_uint64_t = 102;
      // uses atomic64 aspect
      auto ref_uint64 =
          sycl::atomic_ref<uint64_t, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(
              val_uint64_t);
      float val_float = 103.0;
      // doesn't use atomic64 aspect
      auto ref_float =
          sycl::atomic_ref<float, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(val_float);
      int val_int = 104;
      // doesn't use atomic64 aspect
      auto ref_int =
          sycl::atomic_ref<int, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(val_int);

      double *ptr = nullptr;
      auto ref_double_ptr =
          sycl::atomic_ref<double *, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(ptr);
    });
  });
  return 0;
}

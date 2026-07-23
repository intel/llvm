// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64-unknown-unknown -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s

// This test checks that compile-time kernel properties attached to a free
// function kernel (e.g. sub_group_size / work_group_size) are propagated to the
// kernel generated for the free function kernel enqueue functions (nd_launch /
// single_task taking a kernel_function_s). Those enqueue functions submit a
// helper "wrapper" kernel (NdRangeFreeFunctionKernelWrapper /
// SingleTaskFreeFunctionKernelWrapper) whose call operator merely forwards to
// the free function kernel, so the properties have to be copied onto the
// wrapper kernel explicitly. The free-function-kernel-kind markers
// (sycl-nd-range-kernel / sycl-single-task-kernel), on the other hand, must NOT
// be copied onto the wrapper kernel.

#include "sycl.hpp"

using namespace sycl;

// Minimal reproduction of the enqueue wrapper types declared by
// sycl/ext/oneapi/experimental/enqueue_functions.hpp. Only the names and the
// first (function pointer) template argument matter for the Sema logic under
// test.
namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
template <auto *Func, int Dimensions, int tag, typename... ArgsT>
struct NdRangeFreeFunctionKernelWrapper;
template <auto *Func, int tag, typename... ArgsT>
struct SingleTaskFreeFunctionKernelWrapper;
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

namespace syclexp = sycl::ext::oneapi::experimental;

// Free function kernel with both a kind marker (nd-range) and a compile-time
// property (sub-group-size).
[[__sycl_detail__::add_ir_attributes_function(
    "sycl-nd-range-kernel", "sycl-sub-group-size", 1, 16)]]
void ndr_ff(int *ptr) { ptr[0] = 1; }

// Free function kernel with a kind marker (single-task) and a compile-time
// property (work-group-size).
[[__sycl_detail__::add_ir_attributes_function(
    "sycl-single-task-kernel", "sycl-work-group-size", 0, "8")]]
void stk_ff(int *ptr) { ptr[0] = 2; }

// Free function kernel with only a kind marker and no compile-time properties.
// The generated wrapper kernel must not gain any property attributes and, in
// particular, must not gain the kind marker.
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void ndr_ff_no_props(int *ptr) { ptr[0] = 3; }

queue q;

void launch() {
  q.submit([&](handler &h) {
    int *ptr = nullptr;
    h.parallel_for<
        syclexp::detail::NdRangeFreeFunctionKernelWrapper<&ndr_ff, 1, 2,
                                                           int *>>(
        range<1>{1}, [=](id<1>) { ndr_ff(ptr); });
  });
  q.submit([&](handler &h) {
    int *ptr = nullptr;
    h.single_task<
        syclexp::detail::SingleTaskFreeFunctionKernelWrapper<&stk_ff, 2,
                                                             int *>>(
        [=]() { stk_ff(ptr); });
  });
  q.submit([&](handler &h) {
    int *ptr = nullptr;
    h.parallel_for<
        syclexp::detail::NdRangeFreeFunctionKernelWrapper<&ndr_ff_no_props, 1, 2,
                                                          int *>>(
        range<1>{1}, [=](id<1>) { ndr_ff_no_props(ptr); });
  });
}

// The nd-range wrapper kernel gets the sub-group-size property, materialized as
// intel_reqd_sub_group_size metadata (previously dropped, this is the fix).
// CHECK: define {{.*}}spir_kernel void @{{.*}}NdRangeFreeFunctionKernelWrapper{{.*}}ndr_ff{{[^_]}}{{.*}}(ptr addrspace(1) {{[^)]*}}) #[[NDR_ATTR:[0-9]+]]{{.*}}!intel_reqd_sub_group_size ![[SGSIZE:[0-9]+]]

// The single-task wrapper kernel gets the work-group-size property, materialized
// as reqd_work_group_size metadata (previously dropped, this is the fix).
// CHECK: define {{.*}}spir_kernel void @{{.*}}SingleTaskFreeFunctionKernelWrapper{{.*}}stk_ff{{.*}}(ptr addrspace(1) {{[^)]*}}) #[[STK_ATTR:[0-9]+]]{{.*}}!reqd_work_group_size ![[WGSIZE:[0-9]+]]

// The wrapper for the free function kernel without properties gets no property
// metadata at all.
// CHECK: define {{.*}}spir_kernel void @{{.*}}NdRangeFreeFunctionKernelWrapper{{.*}}ndr_ff_no_props{{.*}}(ptr addrspace(1) {{[^)]*}}) #[[NOPROP_ATTR:[0-9]+]] {{[^!]*}}!sycl_fixed_targets

// Full attribute-group matches. Because the SYCL string attributes are emitted
// in sorted order, matching the complete list proves both that the compile-time
// property is present AND that the free-function-kernel-kind marker
// (sycl-nd-range-kernel / sycl-single-task-kernel) was NOT copied onto the
// wrapper kernel.
// CHECK-DAG: attributes #[[NDR_ATTR]] = { convergent {{.*}}"sycl-module-id"={{[^ ]+}} "sycl-optlevel"="0" "sycl-sub-group-size"="16" "uniform-work-group-size" }
// CHECK-DAG: attributes #[[STK_ATTR]] = { convergent {{.*}}"sycl-module-id"={{[^ ]+}} "sycl-optlevel"="0" "sycl-work-group-size"="8" "uniform-work-group-size" }
// CHECK-DAG: attributes #[[NOPROP_ATTR]] = { convergent {{.*}}"sycl-module-id"={{[^ ]+}} "sycl-optlevel"="0" "uniform-work-group-size" }

// CHECK-DAG: ![[SGSIZE]] = !{i32 16}
// CHECK-DAG: ![[WGSIZE]] = !{i64 8, i64 1, i64 1}

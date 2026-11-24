// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s
//
// Checks that the kernel properties embedded via
// get(sycl::khr::property::tag::kernel) are lowered to the same IR attributes
// as the equivalent sycl_ext_oneapi_kernel_properties properties, for each
// kind of kernel invocation.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

struct NDRangeKernel {
  void operator()(sycl::nd_item<2>) const {}
  auto get(kp::tag::kernel) const {
    return properties{kp::work_group_size<8, 4>, kp::work_group_size_hint<2, 2>,
                      kp::sub_group_size<16>};
  }
};

struct SingleTaskKernel {
  void operator()() const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<1>; }
};

struct RangeKernel {
  void operator()(sycl::item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<1, 2, 3>; }
};

struct ReductionKernel {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) const { return kp::sub_group_size<32>; }
};

struct ReductionPropsKernel {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
};

void test(sycl::queue &q, int *sum) {
  q.submit([&](sycl::handler &cgh) {
    // CHECK-DAG: spir_kernel void @{{.*}}NDRangeKernel{{.*}} #[[NDRangeAttrs:[0-9]+]]
    cgh.parallel_for<class NDRangeKernel>(sycl::nd_range<2>{{16, 16}, {8, 4}},
                                          NDRangeKernel{});
    // CHECK-DAG: spir_kernel void @{{.*}}SingleTaskKernel{{.*}} #[[SingleTaskAttrs:[0-9]+]]
    cgh.single_task<class SingleTaskKernel>(SingleTaskKernel{});
    // Range rounding generates a second kernel wrapping the function object;
    // both get the property.
    // CHECK-DAG: spir_kernel void @_ZTS11RangeKernel() {{.*}} #[[RangeAttrs:[0-9]+]]
    // CHECK-DAG: spir_kernel void @{{.*}}RoundedRangeKernel{{.*}}RangeKernel{{.*}} #[[RoundedRangeAttrs:[0-9]+]]
    cgh.parallel_for<class RangeKernel>(sycl::range<1>{1000}, RangeKernel{});
    // CHECK-DAG: spir_kernel void @{{.*}}ReductionKernel{{.*}} #[[ReductionAttrs:[0-9]+]]
    cgh.parallel_for<class ReductionKernel>(
        sycl::nd_range<1>{1024, 64}, sycl::reduction(sum, sycl::plus<int>()),
        ReductionKernel{});
    // The embedded properties are merged with the properties passed alongside
    // the reductions.
    // CHECK-DAG: spir_kernel void @{{.*}}ReductionPropsKernel{{.*}} #[[ReductionPropsAttrs:[0-9]+]]
    cgh.parallel_for<class ReductionPropsKernel>(
        sycl::nd_range<1>{1024, 64},
        sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::sub_group_size<16>},
        sycl::reduction(sum, sycl::plus<int>()), ReductionPropsKernel{});
  });
}

// CHECK-DAG: attributes #[[NDRangeAttrs]] = { {{.*}}"sycl-sub-group-size"="16"{{.*}}"sycl-work-group-size"="8,4"{{.*}}"sycl-work-group-size-hint"="2,2"
// CHECK-DAG: attributes #[[SingleTaskAttrs]] = { {{.*}}"sycl-single-task"{{.*}}"sycl-work-group-size"="1"
// CHECK-DAG: attributes #[[RangeAttrs]] = { {{.*}}"sycl-work-group-size"="1,2,3"
// CHECK-DAG: attributes #[[RoundedRangeAttrs]] = { {{.*}}"sycl-work-group-size"="1,2,3"
// CHECK-DAG: attributes #[[ReductionAttrs]] = { {{.*}}"sycl-sub-group-size"="32"
// CHECK-DAG: attributes #[[ReductionPropsAttrs]] = { {{.*}}"sycl-sub-group-size"="16"{{.*}}"sycl-work-group-size"="64"

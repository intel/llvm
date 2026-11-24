// RUN: %clangxx -fsycl-device-only -fsyntax-only -Xclang -verify %s
//
// Checks the device compiler diagnostics for kernel properties embedded via
// get(sycl::khr::property::tag::kernel) that conflict with kernel attributes
// on the function call operator.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

struct AttributeMismatch {
  // expected-error@+3 {{'sycl::reqd_work_group_size' attribute arguments '32' do not match the value '64' of kernel property 'work_group_size'}}
  // expected-error@+3 {{'sycl::work_group_size_hint' attribute arguments '64,1' do not match the value '64' of kernel property 'work_group_size_hint'}}
  // expected-error@+3 {{'sycl::reqd_sub_group_size' attribute arguments '8' do not match the value '16' of kernel property 'sub_group_size'}}
  [[sycl::reqd_work_group_size(32)]]
  [[sycl::work_group_size_hint(64, 1)]]
  [[sycl::reqd_sub_group_size(8)]] void operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const {
    return properties{kp::work_group_size<64>, kp::work_group_size_hint<64>,
                      kp::sub_group_size<16>};
  }
};

struct AttributeMatch {
  [[sycl::reqd_work_group_size(64)]] [[sycl::reqd_sub_group_size(16)]] void
  operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const {
    return properties{kp::work_group_size<64>, kp::sub_group_size<16>};
  }
};

// An attribute without an equivalent property is not checked.
struct AttributeOnly {
  [[sycl::reqd_work_group_size(32)]] void operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::sub_group_size<16>; }
};

// The getter is forwarded by the range rounding wrapper.
struct RangeKernel {
  // expected-error@+1 {{'sycl::reqd_work_group_size' attribute arguments '16' do not match the value '8' of kernel property 'work_group_size'}}
  [[sycl::reqd_work_group_size(16)]] void operator()(sycl::item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<8>; }
};

void test(sycl::queue &q) {
  q.parallel_for(sycl::nd_range<1>{1024, 64}, AttributeMismatch{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, AttributeMatch{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, AttributeOnly{});
  q.parallel_for(sycl::range<1>{1000}, RangeKernel{});
}

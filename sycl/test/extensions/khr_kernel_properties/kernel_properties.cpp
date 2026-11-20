// The device compiler warns about kernels with both kernel attributes and
// kernel properties (see the MatchingAttribute kernel).
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DSPLIT_HEADERS %s
//
// Checks the sycl_khr_kernel_properties API: the feature test macro, the
// kernel tag type, the properties and their traits, and that function objects
// embedding them are accepted by the kernel invocation functions.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
// expected-no-diagnostics
#ifdef SPLIT_HEADERS
#include <sycl/khr/kernel_properties.hpp>
#include <sycl/khr/split_headers/version.hpp>
#else
#include <sycl/sycl.hpp>
#endif

#ifndef SYCL_KHR_KERNEL_PROPERTIES
#error "SYCL_KHR_KERNEL_PROPERTIES is not defined"
#endif
static_assert(SYCL_KHR_KERNEL_PROPERTIES == 1);

namespace kp = sycl::khr::property;
using namespace sycl::khr;

static_assert(std::is_class_v<kp::tag::kernel>);

template <typename P, typename K> constexpr bool checkProperty() {
  static_assert(is_property_v<P>);
  static_assert(is_property_key_v<K>);
  static_assert(is_property_key_compile_time_v<K>);
  static_assert(is_property_for_v<P, kp::tag::kernel>);
  static_assert(is_property_key_for_v<K, kp::tag::kernel>);
  static_assert(std::is_trivially_copyable_v<P>);
  return true;
}

static_assert(checkProperty<decltype(kp::work_group_size<8>),
                            kp::key::work_group_size>());
static_assert(checkProperty<decltype(kp::work_group_size<8, 4, 2>),
                            kp::key::work_group_size>());
static_assert(checkProperty<decltype(kp::work_group_size_hint<1, 2>),
                            kp::key::work_group_size_hint>());
static_assert(checkProperty<decltype(kp::sub_group_size<16>),
                            kp::key::sub_group_size>());

static_assert(std::is_same_v<std::remove_cv_t<decltype(
                                 decltype(kp::work_group_size<8, 4, 2>)::value)>,
                             std::size_t[3]>);
static_assert(decltype(kp::work_group_size<8, 4, 2>)::value[0] == 8);
static_assert(decltype(kp::work_group_size<8, 4, 2>)::value[2] == 2);
static_assert(decltype(kp::work_group_size_hint<1, 2>)::value[1] == 2);
static_assert(
    std::is_same_v<decltype(decltype(kp::sub_group_size<16>)::value),
                   const std::uint32_t>);
static_assert(decltype(kp::sub_group_size<16>)::value == 16);

// Not properties for other classes.
struct OtherClass {};
static_assert(!is_property_for_v<decltype(kp::work_group_size<8>), OtherClass>);

constexpr auto KernelProps =
    properties{kp::work_group_size<64>, kp::work_group_size_hint<64>,
               kp::sub_group_size<16>};
using KernelPropsT = std::remove_cv_t<decltype(KernelProps)>;
static_assert(is_property_or_list_for_v<KernelPropsT, kp::tag::kernel>);
static_assert(KernelPropsT::has_property<kp::key::work_group_size>());

#ifndef SPLIT_HEADERS
template <std::size_t WG, std::uint32_t SG> struct Scale {
  float *data;
  void operator()(sycl::nd_item<1> it) const {
    data[it.get_global_linear_id()] *= 2.0f;
  }
  auto get(kp::tag::kernel) const {
    return properties{kp::work_group_size<WG>, kp::sub_group_size<SG>};
  }
};

struct SingleProperty {
  void operator()() const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<1>; }
};

struct RangeKernel {
  void operator()(sycl::item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size_hint<8>; }
};

struct EmptyList {
  void operator()() const {}
  auto get(kp::tag::kernel) const { return properties{}; }
};

// Matching attribute and property: no error.
struct MatchingAttribute {
  [[sycl::reqd_work_group_size(64)]] void operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
};

// Both getters.
struct BothGetters {
  void operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size_hint<64>};
  }
};

struct ReductionKernel {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
};

void test(sycl::queue &q, float *data, int *sum) {
  q.parallel_for(sycl::nd_range<1>{1024, 64}, Scale<64, 16>{data});
  q.single_task(SingleProperty{});
  q.parallel_for(sycl::range<1>{1024}, RangeKernel{});
  q.single_task(EmptyList{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, MatchingAttribute{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, BothGetters{});
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>{1024, 64}, Scale<64, 16>{data});
    cgh.parallel_for(sycl::nd_range<1>{1024, 64},
                     sycl::reduction(sum, sycl::plus<int>()),
                     ReductionKernel{});
  });
}
#endif

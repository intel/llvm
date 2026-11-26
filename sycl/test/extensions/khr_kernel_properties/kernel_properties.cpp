// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DSPLIT_HEADERS %s
//
// Checks the sycl_khr_kernel_properties API: the feature test macro, the
// kernel tag type, the properties and their traits, and that function objects
// embedding them are accepted by the kernel invocation functions. The split
// headers mode also checks the kernel properties against the queue and
// reduction properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
// expected-no-diagnostics
#ifdef SPLIT_HEADERS
#include <sycl/khr/kernel_properties.hpp>
#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/queue.hpp>
#include <sycl/khr/split_headers/reduction.hpp>
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

// Unlike the other tags, `kernel` is a complete type usable as a parameter.
static_assert(std::is_class_v<kp::tag::kernel>);
static_assert(std::is_empty_v<kp::tag::kernel>);
static_assert(std::is_trivially_default_constructible_v<kp::tag::kernel>);
static_assert(std::is_trivially_copyable_v<kp::tag::kernel>);
constexpr kp::tag::kernel KernelTag{};

template <typename P, typename K> constexpr bool checkProperty() {
  static_assert(is_property_v<P>);
  static_assert(is_property_key_v<K>);
  static_assert(is_property_key_compile_time_v<K>);
  static_assert(is_property_for_v<P, kp::tag::kernel>);
  static_assert(is_property_key_for_v<K, kp::tag::kernel>);
  static_assert(is_property_or_list_for_v<P, kp::tag::kernel>);
  static_assert(std::is_trivially_copyable_v<P>);

  static_assert(std::is_base_of_v<std::true_type, is_property<P>>);
  static_assert(std::is_base_of_v<std::true_type, is_property_key<K>>);
  static_assert(
      std::is_base_of_v<std::true_type, is_property_key_compile_time<K>>);
  static_assert(
      std::is_base_of_v<std::true_type, is_property_for<P, kp::tag::kernel>>);
  static_assert(std::is_base_of_v<std::true_type,
                                  is_property_key_for<K, kp::tag::kernel>>);
  static_assert(std::is_base_of_v<std::true_type,
                                  is_property_or_list_for<P, kp::tag::kernel>>);

  static_assert(!is_property_v<K>);
  static_assert(!is_property_key_v<P>);
  static_assert(!is_property_for_v<K, kp::tag::kernel>);
  static_assert(!is_property_key_for_v<P, kp::tag::kernel>);

  static_assert(!is_property_for_v<P, sycl::queue>);
  static_assert(!is_property_key_for_v<K, sycl::queue>);
  static_assert(!is_property_or_list_for_v<P, sycl::queue>);
  static_assert(!is_property_for_v<P, kp::tag::reduction>);
  static_assert(!is_property_key_for_v<K, kp::tag::reduction>);
  static_assert(!std::is_constructible_v<sycl::queue, P>);
  static_assert(!std::is_constructible_v<sycl::queue, properties<P>>);
  static_assert(!std::is_constructible_v<sycl::queue, const sycl::device &, P>);

  // The static `get_property` is usable in constant expressions.
  constexpr auto Prop = properties<P>::template get_property<K>();
  static_assert(
      std::is_same_v<std::remove_cv_t<decltype(Prop)>, std::remove_cv_t<P>>);
  return true;
}

static_assert(checkProperty<decltype(kp::work_group_size<8>),
                            kp::key::work_group_size>());
static_assert(checkProperty<decltype(kp::work_group_size<8, 4>),
                            kp::key::work_group_size>());
static_assert(checkProperty<decltype(kp::work_group_size<8, 4, 2>),
                            kp::key::work_group_size>());
static_assert(checkProperty<decltype(kp::work_group_size_hint<8>),
                            kp::key::work_group_size_hint>());
static_assert(checkProperty<decltype(kp::work_group_size_hint<1, 2>),
                            kp::key::work_group_size_hint>());
static_assert(checkProperty<decltype(kp::work_group_size_hint<8, 4, 2>),
                            kp::key::work_group_size_hint>());
static_assert(
    checkProperty<decltype(kp::sub_group_size<16>), kp::key::sub_group_size>());

static_assert(
    std::is_same_v<std::remove_cv_t<
                       decltype(decltype(kp::work_group_size<8, 4, 2>)::value)>,
                   std::size_t[3]>);
static_assert(decltype(kp::work_group_size<8, 4, 2>)::value[0] == 8);
static_assert(decltype(kp::work_group_size<8, 4, 2>)::value[2] == 2);
static_assert(decltype(kp::work_group_size_hint<1, 2>)::value[1] == 2);
static_assert(std::is_same_v<decltype(decltype(kp::sub_group_size<16>)::value),
                             const std::uint32_t>);
static_assert(decltype(kp::sub_group_size<16>)::value == 16);
static_assert(
    std::is_same_v<std::remove_cv_t<
                       decltype(decltype(kp::work_group_size_hint<8>)::value)>,
                   std::size_t[1]>);
// `value` is also readable through the variables.
static_assert(kp::work_group_size<8, 4, 2>.value[1] == 4);
static_assert(kp::work_group_size_hint<8, 4, 2>.value[2] == 2);
static_assert(kp::sub_group_size<32>.value == 32);

static_assert(!std::is_same_v<decltype(kp::work_group_size<8>),
                              decltype(kp::work_group_size<4>)>);
static_assert(!std::is_same_v<decltype(kp::work_group_size<8>),
                              decltype(kp::work_group_size<8, 1>)>);
static_assert(!std::is_same_v<decltype(kp::sub_group_size<8>),
                              decltype(kp::sub_group_size<16>)>);
static_assert(
    !std::is_same_v<std::remove_cv_t<decltype(kp::work_group_size<8>)>,
                    std::remove_cv_t<decltype(kp::work_group_size_hint<8>)>>);
static_assert(
    !std::is_same_v<kp::key::work_group_size, kp::key::work_group_size_hint>);

// Not properties for other classes.
struct OtherClass {};
static_assert(!is_property_for_v<decltype(kp::work_group_size<8>), OtherClass>);
static_assert(!is_property_key_for_v<kp::key::sub_group_size, OtherClass>);

// Other properties, and other types, are not kernel properties.
static_assert(!is_property_for_v<kp::in_order, kp::tag::kernel>);
static_assert(!is_property_for_v<kp::enable_profiling, kp::tag::kernel>);
static_assert(!is_property_for_v<kp::initialize_to_identity, kp::tag::kernel>);
static_assert(!is_property_key_for_v<kp::key::in_order, kp::tag::kernel>);
static_assert(!is_property_or_list_for_v<int, kp::tag::kernel>);
static_assert(!is_property_or_list_for_v<OtherClass, kp::tag::kernel>);
static_assert(!is_property_or_list_for_v<kp::tag::kernel, kp::tag::kernel>);

constexpr auto KernelProps =
    properties{kp::work_group_size<64>, kp::work_group_size_hint<64>,
               kp::sub_group_size<16>};
using KernelPropsT = std::remove_cv_t<decltype(KernelProps)>;
static_assert(is_property_or_list_for_v<KernelPropsT, kp::tag::kernel>);
static_assert(KernelPropsT::has_property<kp::key::work_group_size>());
static_assert(KernelPropsT::has_property<kp::key::work_group_size_hint>());
static_assert(KernelPropsT::has_property<kp::key::sub_group_size>());
static_assert(!is_property_or_list_for_v<KernelPropsT, sycl::queue>);
static_assert(!std::is_constructible_v<sycl::queue, KernelPropsT>);
static_assert(is_property_or_list_for_v<empty_properties_t, kp::tag::kernel>);
static_assert(!is_property_or_list_for_v<
              decltype(properties{kp::work_group_size<8>, kp::in_order{}}),
              kp::tag::kernel>);

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

struct ReductionKernel {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
};

// A declared-only `get` returning the type of a property variable, which is
// const.
struct DeclaredGet {
  void operator()(sycl::nd_item<1>) const {}
  decltype(kp::work_group_size<64>) get(kp::tag::kernel) const;
};

struct DeclaredListGet {
  void operator()(sycl::item<1>) const {}
  decltype(properties{kp::work_group_size_hint<8>,
                      kp::sub_group_size<16>}) get(kp::tag::kernel) const;
};

void test(sycl::queue &q, float *data, int *sum) {
  q.parallel_for(sycl::nd_range<1>{1024, 64}, Scale<64, 16>{data});
  q.single_task(SingleProperty{});
  q.parallel_for(sycl::range<1>{1024}, RangeKernel{});
  q.single_task(EmptyList{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, MatchingAttribute{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, DeclaredGet{});
  q.parallel_for(sycl::range<1>{1024}, DeclaredListGet{});
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>{1024, 64}, Scale<64, 16>{data});
    cgh.parallel_for(sycl::nd_range<1>{1024, 64},
                     sycl::reduction(sum, sycl::plus<int>()),
                     ReductionKernel{});
  });
}
#endif

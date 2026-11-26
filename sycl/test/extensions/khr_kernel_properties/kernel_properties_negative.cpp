// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note -D__SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__ %s
//
// Checks the diagnostics issued by the SYCL headers for kernel properties
// embedded via get(sycl::khr::property::tag::kernel), including function
// objects that also have a get(sycl::ext::oneapi::experimental::properties_tag)
// member, and the mandates of the properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

struct NonConstGetter {
  void operator()() const {}
  auto get(kp::tag::kernel) { return kp::work_group_size<1>; }
};

struct NotAKernelProperty {
  void operator()() const {}
  auto get(kp::tag::kernel) const { return kp::in_order{}; }
};

struct NotAProperty {
  void operator()() const {}
  int get(kp::tag::kernel) const { return 0; }
};

struct MixedList {
  void operator()() const {}
  auto get(kp::tag::kernel) const {
    return properties{kp::work_group_size<1>, kp::in_order{}};
  }
};

struct ReductionNonConstGetter {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) { return kp::work_group_size<64>; }
};

struct ReductionBothGetters {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<32>};
  }
};

struct BothGetters {
  void operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::sub_group_size<16>; }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<32>};
  }
};

// Range rounding wraps the function object; each of these is diagnosed once,
// also when only the wrapped kernel is generated. The template parameter
// distinguishes the queue and handler invocations.
template <int> struct RangeNonConstGetter {
  void operator()(sycl::item<1>) const {}
  auto get(kp::tag::kernel) { return kp::work_group_size<8>; }
};

template <int> struct RangeNotAProperty {
  void operator()(sycl::item<1>) const {}
  int get(kp::tag::kernel) const { return 0; }
};

template <int> struct RangeBothGetters {
  void operator()(sycl::item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<8>; }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<16>};
  }
};

// Even with the same values.
struct ReductionConflict {
  template <typename T> void operator()(sycl::nd_item<1>, T &sum) const {
    sum += 1;
  }
  auto get(kp::tag::kernel) const { return kp::work_group_size<64>; }
};

struct BothGettersSame {
  void operator()(sycl::nd_item<1>) const {}
  auto get(kp::tag::kernel) const { return kp::sub_group_size<16>; }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>};
  }
};

void test(sycl::queue &q, int *sum) {
  // expected-error-re@*:* 5 {{static assertion failed due to requirement {{.*}}A kernel function object must not embed kernel properties via both get(sycl::ext::oneapi::experimental::properties_tag) and get(sycl::khr::property::tag::kernel).}}
  q.parallel_for(sycl::nd_range<1>{1024, 64}, BothGetters{});
  q.parallel_for(sycl::range<1>{1000}, RangeBothGetters<0>{});
  q.parallel_for(sycl::nd_range<1>{1024, 64}, BothGettersSame{});
  // expected-error-re@*:* 4 {{static assertion failed due to requirement {{.*}}get(sycl::khr::property::tag::kernel) member in kernel function object must be declared as a const member function.}}
  q.single_task(NonConstGetter{});
  q.parallel_for(sycl::range<1>{1000}, RangeNonConstGetter<0>{});
  // expected-error-re@*:* 5 {{static assertion failed due to requirement {{.*}}The return type of get(sycl::khr::property::tag::kernel) must be a property or a property list{{.*}}}}
  q.single_task(NotAKernelProperty{});
  q.single_task(NotAProperty{});
  q.single_task(MixedList{});
  q.parallel_for(sycl::range<1>{1000}, RangeNotAProperty<0>{});
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{1000}, RangeBothGetters<1>{});
    cgh.parallel_for(sycl::range<1>{1000}, RangeNonConstGetter<1>{});
    cgh.parallel_for(sycl::range<1>{1000}, RangeNotAProperty<1>{});
    cgh.parallel_for(sycl::nd_range<1>{1024, 64},
                     sycl::reduction(sum, sycl::plus<int>()),
                     ReductionNonConstGetter{});
    cgh.parallel_for(sycl::nd_range<1>{1024, 64},
                     sycl::reduction(sum, sycl::plus<int>()),
                     ReductionBothGetters{});
    // The embedded properties are merged with the properties passed alongside
    // the reductions.
    // expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Failed to merge property lists due to conflicting properties.}}
    // expected-warning-re@+1 2 {{'parallel_for{{.*}}' is deprecated}}
    cgh.parallel_for(sycl::nd_range<1>{1024, 64},
                     sycl::ext::oneapi::experimental::properties{
                         sycl::ext::oneapi::experimental::work_group_size<32>},
                     sycl::reduction(sum, sycl::plus<int>()),
                     ReductionConflict{});
  });
}

// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}sycl::khr::property::work_group_size requires 1, 2 or 3 dimensions.}}
using WG0 = kp::detail::work_group_size_property<>;
WG0 WG0Instance;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}sycl::khr::property::work_group_size requires 1, 2 or 3 dimensions.}}
auto WG4 = kp::work_group_size<1, 1, 1, 1>;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}sycl::khr::property::work_group_size_hint requires 1, 2 or 3 dimensions.}}
auto WGH4 = kp::work_group_size_hint<1, 1, 1, 1>;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}sycl::khr::property::work_group_size must only contain non-zero values.}}
auto WGZero = kp::work_group_size<1, 0>;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}sycl::khr::property::work_group_size_hint must only contain non-zero values.}}
auto WGHZero = kp::work_group_size_hint<0>;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}sycl::khr::property::sub_group_size must be a non-zero value.}}
auto SGZero = kp::sub_group_size<0>;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Duplicate properties in property list.}}
auto Duplicate = properties{kp::work_group_size<1>, kp::work_group_size<2>};
// The invalid property types above also make the property variable templates
// ill formed.
// expected-error@*:* 5 {{must be initialized by a constant expression}}

// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify %s

// Tests the diagnostics for kernel attributes on the call operator that do not
// match the kernel properties embedded by
// 'get(sycl::khr::property::tag::kernel)' (sycl_khr_kernel_properties). The
// SYCL headers pass the properties as elements of the template parameter pack
// of the kernel entry point.

#include "sycl.hpp"

namespace sycl {
inline namespace _V1 {
namespace khr {
template <typename... Ps> struct properties {};
namespace property {
namespace tag {
struct kernel {};
} // namespace tag
template <int... Dims> struct work_group_size {};
template <int... Dims> struct work_group_size_hint {};
template <int Size> struct sub_group_size {};
} // namespace property
} // namespace khr
} // namespace _V1
} // namespace sycl

namespace kp = sycl::khr::property;
// Mimics PropertyMetaInfo.
template <typename P> struct MetaInfo;
template <> struct MetaInfo<kp::work_group_size<8>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value = "8";
};
template <> struct MetaInfo<kp::work_group_size<8, 4>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value = "8,4";
};
template <> struct MetaInfo<kp::work_group_size_hint<2, 2, 2>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value = "2,2,2";
};
template <> struct MetaInfo<kp::sub_group_size<16>> {
  static constexpr const char *name = "sycl-sub-group-size";
  static constexpr int value = 16;
};
template <typename KernelName, typename KernelType, typename... Props>
[[__sycl_detail__::add_ir_attributes_function(
    "sycl-single-task", MetaInfo<Props>::name..., nullptr,
    MetaInfo<Props>::value...)]] __attribute__((sycl_kernel)) void
kernel_single_task(const KernelType &KernelFunc) {
  KernelFunc();
}

template <typename KernelName, typename KernelType, typename... Props>
[[__sycl_detail__::add_ir_attributes_function(
    MetaInfo<Props>::name..., MetaInfo<Props>::value...)]]
__attribute__((sycl_kernel)) void
kernel_parallel_for(const KernelType &KernelFunc) {
  KernelFunc();
}

// Matching attributes: no error. The existing warning about kernels with both
// attributes and properties is unaffected.
struct Matching {
  // expected-warning@+3 {{kernel has both attribute 'sycl::reqd_work_group_size' and kernel properties}}
  // expected-warning@+2 {{kernel has both attribute 'sycl::work_group_size_hint' and kernel properties}}
  // expected-warning@+2 {{kernel has both attribute 'sycl::reqd_sub_group_size' and kernel properties}}
  [[sycl::reqd_work_group_size(8, 4)]] [[sycl::work_group_size_hint(2, 2, 2)]]
  [[sycl::reqd_sub_group_size(16)]] void operator()() const {}
  auto get(kp::tag::kernel) const {
    return sycl::khr::properties<kp::work_group_size<8, 4>,
                                 kp::work_group_size_hint<2, 2, 2>,
                                 kp::sub_group_size<16>>{};
  }
};

// A single property instead of a list.
struct MatchingSingle {
  // expected-warning@+1 {{kernel has both attribute 'sycl::reqd_work_group_size' and kernel properties}}
  [[sycl::reqd_work_group_size(8)]] void operator()() const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<8>{}; }
};

struct Mismatching {
  // expected-error@+6 {{'sycl::reqd_work_group_size' attribute arguments '8' do not match the value '8,4' of kernel property 'sycl::khr::property::work_group_size' embedded by 'get(sycl::khr::property::tag::kernel)'}}
  // expected-warning@+5 {{kernel has both attribute 'sycl::reqd_work_group_size' and kernel properties}}
  // expected-error@+5 {{'sycl::work_group_size_hint' attribute arguments '2,2' do not match the value '2,2,2' of kernel property 'sycl::khr::property::work_group_size_hint' embedded by 'get(sycl::khr::property::tag::kernel)'}}
  // expected-warning@+4 {{kernel has both attribute 'sycl::work_group_size_hint' and kernel properties}}
  // expected-error@+4 {{'sycl::reqd_sub_group_size' attribute arguments '32' do not match the value '16' of kernel property 'sycl::khr::property::sub_group_size' embedded by 'get(sycl::khr::property::tag::kernel)'}}
  // expected-warning@+3 {{kernel has both attribute 'sycl::reqd_sub_group_size' and kernel properties}}
  [[sycl::reqd_work_group_size(8)]]
  [[sycl::work_group_size_hint(2, 2)]]
  [[sycl::reqd_sub_group_size(32)]] void operator()() const {}
  // expected-note@+1 3 {{kernel properties are embedded by this member function}}
  auto get(kp::tag::kernel) const {
    return sycl::khr::properties<kp::work_group_size<8, 4>,
                                 kp::work_group_size_hint<2, 2, 2>,
                                 kp::sub_group_size<16>>{};
  }
};

// A property in the pack that does not come from the khr getter is not
// checked against the attributes.
struct OtherProperty {};
template <> struct MetaInfo<OtherProperty> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value = "16";
};
struct NotFromGetter {
  // expected-warning@+1 {{kernel has both attribute 'sycl::reqd_work_group_size' and kernel properties}}
  [[sycl::reqd_work_group_size(8)]] void operator()() const {}
  auto get(kp::tag::kernel) const { return kp::work_group_size<8>{}; }
};

// The getter is inherited from a base class.
struct Base {
  // expected-note@+1 {{kernel properties are embedded by this member function}}
  auto get(kp::tag::kernel) const { return kp::sub_group_size<16>{}; }
};
struct Derived : Base {
  // expected-error@+2 {{'sycl::reqd_sub_group_size' attribute arguments '8' do not match the value '16' of kernel property 'sycl::khr::property::sub_group_size' embedded by 'get(sycl::khr::property::tag::kernel)'}}
  // expected-warning@+1 {{kernel has both attribute 'sycl::reqd_sub_group_size' and kernel properties}}
  [[sycl::reqd_sub_group_size(8)]] void operator()() const {}
};

// Not the khr tag: the khr checks do not apply.
namespace other {
struct kernel {};
}
struct WrongTag {
  // expected-warning@+1 {{kernel has both attribute 'sycl::reqd_work_group_size' and kernel properties}}
  [[sycl::reqd_work_group_size(4)]] void operator()() const {}
  auto get(other::kernel) const { return kp::work_group_size<8>{}; }
};

// A non-const getter is not used (the SYCL headers reject it).
struct NonConstGetter {
  // expected-warning@+1 {{kernel has both attribute 'sycl::reqd_work_group_size' and kernel properties}}
  [[sycl::reqd_work_group_size(4)]] void operator()() const {}
  auto get(kp::tag::kernel) { return kp::work_group_size<8>{}; }
};

int main() {
  kernel_single_task<class K1, Matching, kp::work_group_size<8, 4>,
                     kp::work_group_size_hint<2, 2, 2>, kp::sub_group_size<16>>(
      Matching{});
  kernel_parallel_for<class K2, MatchingSingle, kp::work_group_size<8>>(
      MatchingSingle{});
  kernel_single_task<class K3, Mismatching, kp::work_group_size<8, 4>,
                     kp::work_group_size_hint<2, 2, 2>, kp::sub_group_size<16>>(
      Mismatching{});
  kernel_parallel_for<class K4, NotFromGetter, OtherProperty,
                      kp::work_group_size<8>>(NotFromGetter{});
  kernel_single_task<class K7, Derived, kp::sub_group_size<16>>(Derived{});
  kernel_single_task<class K8, WrongTag, kp::work_group_size<8>>(WrongTag{});
  kernel_single_task<class K9, NonConstGetter, kp::work_group_size<8>>(
      NonConstGetter{});
}

// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Tests the sycl_khr_properties core: classification traits, the properties
// container (has_property/get_property/CTAD/empty), the feature-test macro, and
// the convenience base classes for defining runtime, compile-time, and hybrid
// properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/khr/properties.hpp>

#ifndef SYCL_KHR_PROPERTIES
#error "SYCL_KHR_PROPERTIES feature-test macro is not defined"
#endif

namespace kd = sycl::khr::detail;
using namespace sycl::khr;

struct MyClass {};
struct OtherClass {};

// Runtime property (convenience base + separate key).
struct enable_profiling_key : kd::runtime_property_key {};
struct enable_profiling : kd::runtime_property<enable_profiling_key> {
  bool value;
  constexpr enable_profiling(bool v = true) : value(v) {}
};
template <>
struct sycl::khr::is_property_key_for<enable_profiling_key, MyClass>
    : std::true_type {};

// Compile-time property with a single non-type value (convenience base).
struct alignment_key : kd::constant_value_property_key {};
template <int A>
inline constexpr alignment_key::__detail_property_t<alignment_key, int, A>
    alignment;
template <>
struct sycl::khr::is_property_key_for<alignment_key, MyClass> : std::true_type {
};

// Compile-time property with a single type value (convenience base).
struct alignment_type_key : kd::constant_type_property_key {};
template <typename T>
inline constexpr alignment_type_key::__detail_property_t<alignment_type_key, T>
    alignment_type;
template <>
struct sycl::khr::is_property_key_for<alignment_type_key, MyClass>
    : std::true_type {};

// Hybrid property (compile-time X, runtime Y) via convenience base.
struct hybrid_key : kd::hybrid_property_key {};
template <int X> struct hybrid : kd::hybrid_property<hybrid_key> {
  static constexpr int x = X;
  int y;
  constexpr hybrid(int y) : y(y) {}
};
template <>
struct sycl::khr::is_property_key_for<hybrid_key, MyClass> : std::true_type {};

// is_property / is_property_key / is_property_key_compile_time.
static_assert(is_property_v<enable_profiling> && is_property_v<hybrid<1>>);
static_assert(
    is_property_v<alignment_key::__detail_property_t<alignment_key, int, 4>>);
static_assert(!is_property_v<int> && !is_property_v<enable_profiling_key>);
static_assert(is_property_key_v<enable_profiling_key> &&
              is_property_key_v<alignment_key> &&
              is_property_key_v<hybrid_key>);
static_assert(!is_property_key_v<enable_profiling>);
static_assert(is_property_key_compile_time_v<alignment_key> &&
              is_property_key_compile_time_v<alignment_type_key>);
// A hybrid key has runtime values, so it is NOT a compile-time key.
static_assert(!is_property_key_compile_time_v<enable_profiling_key> &&
              !is_property_key_compile_time_v<hybrid_key>);

// is_property_key_for / is_property_for / is_property_list_for.
static_assert(is_property_key_for_v<enable_profiling_key, MyClass>);
static_assert(!is_property_key_for_v<enable_profiling_key, OtherClass>);
static_assert(is_property_for_v<enable_profiling, MyClass> &&
              is_property_for_v<hybrid<1>, MyClass>);
static_assert(!is_property_for_v<enable_profiling, OtherClass>);

// Container: CTAD, has_property, get_property (runtime, compile-time, hybrid).
void container() {
  properties p{enable_profiling{true}, alignment<16>, hybrid<3>{4}};
  static_assert(p.has_property<enable_profiling_key>());
  static_assert(p.has_property<alignment_key>());
  static_assert(p.has_property<hybrid_key>());
  static_assert(!p.has_property<alignment_type_key>());
  static_assert(decltype(p)::get_property<alignment_key>().value == 16);
  static_assert(!empty_properties_t::has_property<enable_profiling_key>());

  // is_property_list_for over a populated and the empty list.
  static_assert(is_property_list_for_v<decltype(p), MyClass>);
  static_assert(!is_property_list_for_v<decltype(p), OtherClass>);
  static_assert(is_property_list_for_v<empty_properties_t, OtherClass>);
}

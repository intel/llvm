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

namespace kd = sycl::khr::property::detail;
using namespace sycl::khr;

struct MyClass {};
struct OtherClass {};

// Runtime property (convenience base + separate key).
struct enable_profiling_key : kd::runtime_property_key<enable_profiling_key> {};
struct enable_profiling
    : kd::runtime_property<enable_profiling, enable_profiling_key> {
  bool value;
  constexpr enable_profiling(bool v = true) : value(v) {}
};
template <>
struct sycl::khr::is_property_key_for<enable_profiling_key, MyClass>
    : std::true_type {};

// Compile-time property with a single non-type value (convenience base).
struct alignment_key : kd::constant_value_property_key<alignment_key> {};
template <int A>
inline constexpr alignment_key::__detail_property_t<int, A> alignment;
template <>
struct sycl::khr::is_property_key_for<alignment_key, MyClass> : std::true_type {
};

// Compile-time property with a single type value (convenience base).
struct alignment_type_key : kd::constant_type_property_key<alignment_type_key> {
};
template <typename T>
inline constexpr alignment_type_key::__detail_property_t<T> alignment_type;
template <>
struct sycl::khr::is_property_key_for<alignment_type_key, MyClass>
    : std::true_type {};

// Hybrid property (compile-time X, runtime Y) via convenience base.
struct hybrid_key : kd::hybrid_property_key<hybrid_key> {};
template <int X> struct hybrid : kd::hybrid_property<hybrid<X>, hybrid_key> {
  static constexpr int x = X;
  int y;
  constexpr hybrid(int y) : y(y) {}
};
template <>
struct sycl::khr::is_property_key_for<hybrid_key, MyClass> : std::true_type {};

// Runtime property for OtherClass only (for mixed-class list checks).
struct other_only_key : kd::runtime_property_key<other_only_key> {};
struct other_only : kd::runtime_property<other_only, other_only_key> {
  int value;
  constexpr other_only(int v = 0) : value(v) {}
};
template <>
struct sycl::khr::is_property_key_for<other_only_key, OtherClass>
    : std::true_type {};

// is_property / is_property_key / is_property_key_compile_time.
static_assert(is_property_v<enable_profiling> && is_property_v<hybrid<1>>);
static_assert(is_property_v<alignment_key::__detail_property_t<int, 4>>);
static_assert(!is_property_v<int> && !is_property_v<enable_profiling_key>);
static_assert(is_property_key_v<enable_profiling_key> &&
              is_property_key_v<alignment_key> &&
              is_property_key_v<alignment_type_key> &&
              is_property_key_v<hybrid_key>);
static_assert(!is_property_key_v<enable_profiling>);
static_assert(is_property_key_compile_time_v<alignment_key> &&
              is_property_key_compile_time_v<alignment_type_key>);
// A hybrid key has runtime values, so it is NOT a compile-time key.
static_assert(!is_property_key_compile_time_v<enable_profiling_key> &&
              !is_property_key_compile_time_v<hybrid_key>);

// Only the exact property and key types are classified: not derived classes,
// classes derived from several properties, or property lists.
struct derived_property : enable_profiling {};
struct two_properties : enable_profiling, other_only {};
struct derived_list : properties<enable_profiling> {};
struct derived_key : enable_profiling_key {};
struct derived_ct_key : alignment_key {};
static_assert(!is_property_v<derived_property> &&
              !is_property_v<two_properties> && !is_property_v<derived_list> &&
              !is_property_v<properties<enable_profiling>> &&
              !is_property_v<empty_properties_t>);
static_assert(!is_property_for_v<derived_property, MyClass> &&
              !is_property_for_v<two_properties, MyClass> &&
              !is_property_for_v<derived_list, MyClass> &&
              !is_property_list_for_v<derived_list, MyClass>);
static_assert(!is_property_key_v<derived_key> &&
              !is_property_key_v<derived_ct_key> &&
              !is_property_key_compile_time_v<derived_ct_key>);
static_assert(
    !std::is_constructible_v<properties<enable_profiling>, derived_property>);

// is_property_key_for / is_property_for, for every property kind.
static_assert(is_property_key_for_v<enable_profiling_key, MyClass> &&
              is_property_key_for_v<alignment_key, MyClass> &&
              is_property_key_for_v<alignment_type_key, MyClass> &&
              is_property_key_for_v<hybrid_key, MyClass>);
static_assert(!is_property_key_for_v<enable_profiling_key, OtherClass> &&
              !is_property_key_for_v<alignment_key, OtherClass> &&
              !is_property_key_for_v<alignment_type_key, OtherClass> &&
              !is_property_key_for_v<hybrid_key, OtherClass>);
static_assert(
    is_property_for_v<enable_profiling, MyClass> &&
    is_property_for_v<alignment_key::__detail_property_t<int, 4>, MyClass> &&
    is_property_for_v<alignment_type_key::__detail_property_t<int>, MyClass> &&
    is_property_for_v<hybrid<1>, MyClass>);
static_assert(!is_property_for_v<enable_profiling, OtherClass> &&
              !is_property_for_v<hybrid<1>, OtherClass>);

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

  // A list mixing properties for different classes is a list for neither.
  properties mixed{enable_profiling{true}, other_only{7}};
  static_assert(is_property_for_v<enable_profiling, MyClass> &&
                is_property_for_v<other_only, OtherClass>);
  static_assert(!is_property_list_for_v<decltype(mixed), MyClass>);
  static_assert(!is_property_list_for_v<decltype(mixed), OtherClass>);
}

// The constructor accepts exactly the encoded properties, in any order.
using align16_t = std::remove_const_t<decltype(alignment<16>)>;
using align8_t = std::remove_const_t<decltype(alignment<8>)>;
static_assert(!std::is_constructible_v<empty_properties_t, int>);
static_assert(!std::is_convertible_v<int, empty_properties_t>);
static_assert(
    std::is_constructible_v<properties<enable_profiling>, enable_profiling>);
static_assert(!std::is_constructible_v<properties<enable_profiling>>);
static_assert(!std::is_constructible_v<properties<enable_profiling>,
                                       enable_profiling, enable_profiling>);
static_assert(!std::is_constructible_v<properties<enable_profiling>,
                                       enable_profiling, other_only>);
static_assert(
    !std::is_constructible_v<properties<enable_profiling>, other_only>);
static_assert(!std::is_constructible_v<properties<enable_profiling>, int>);
static_assert(std::is_constructible_v<properties<enable_profiling, other_only>,
                                      other_only, enable_profiling>);
static_assert(std::is_constructible_v<properties<align16_t>, align16_t>);
static_assert(!std::is_constructible_v<properties<align16_t>, align8_t>);
static_assert(!std::is_constructible_v<properties<hybrid<1>>, hybrid<2>>);
static_assert(std::is_copy_constructible_v<properties<enable_profiling>> &&
              std::is_move_constructible_v<properties<enable_profiling>>);
static_assert(std::is_same_v<decltype(properties{properties{alignment<16>}}),
                             properties<align16_t>>);

// has_property requires a key; get_property requires a key in the list.
template <typename List, typename Key, typename = void>
struct can_has : std::false_type {};
template <typename List, typename Key>
struct can_has<List, Key,
               std::void_t<decltype(List::template has_property<Key>())>>
    : std::true_type {};
template <typename List, typename Key, typename = void>
struct can_get : std::false_type {};
template <typename List, typename Key>
struct can_get<List, Key,
               std::void_t<decltype(std::declval<const List &>()
                                        .template get_property<Key>())>>
    : std::true_type {};
template <typename List, typename Key, typename = void>
struct can_get_static : std::false_type {};
template <typename List, typename Key>
struct can_get_static<List, Key,
                      std::void_t<decltype(List::template get_property<Key>())>>
    : std::true_type {};

using rt_list_t = properties<enable_profiling, hybrid<1>>;
using ct_list_t = properties<align16_t>;
static_assert(can_has<rt_list_t, alignment_key>::value);
static_assert(!can_has<rt_list_t, int>::value &&
              !can_has<rt_list_t, enable_profiling>::value &&
              !can_has<empty_properties_t, int>::value);
static_assert(can_get<rt_list_t, enable_profiling_key>::value &&
              can_get<rt_list_t, hybrid_key>::value &&
              can_get_static<ct_list_t, alignment_key>::value);
static_assert(!can_get<rt_list_t, other_only_key>::value &&
              !can_get<rt_list_t, alignment_key>::value &&
              !can_get<rt_list_t, int>::value &&
              !can_get<rt_list_t, enable_profiling>::value);
static_assert(!can_get_static<rt_list_t, alignment_key>::value &&
              !can_get_static<ct_list_t, alignment_type_key>::value &&
              !can_get<ct_list_t, enable_profiling_key>::value);
static_assert(!can_get<empty_properties_t, enable_profiling_key>::value &&
              !can_get_static<empty_properties_t, alignment_key>::value);

// Properties and property lists are trivially copyable (sizes: see
// properties_layout.cpp).
static_assert(std::is_trivially_copyable_v<enable_profiling>);
static_assert(std::is_trivially_copyable_v<hybrid<1>>);
static_assert(std::is_trivially_copyable_v<empty_properties_t>);
static_assert(std::is_trivially_copyable_v<decltype(properties{
                  enable_profiling{true}, alignment<16>})>);
static_assert(std::is_trivially_copyable_v<decltype(properties{
                  hybrid<1>{2}, enable_profiling{true}, alignment<16>})>);

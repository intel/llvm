#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
struct indirectly_callable_key {
  template <typename Set>
  using value_t =
      sycl::ext::oneapi::experimental::property_value<indirectly_callable_key,
                                                      Set>;
};

template <typename Set = void>
inline constexpr indirectly_callable_key::value_t<Set> indirectly_callable;

struct calls_indirectly_key {
  template <typename First = void, typename... SetIds>
  using value_t =
      sycl::ext::oneapi::experimental::property_value<calls_indirectly_key,
                                                      First, SetIds...>;
};

template <typename First = void, typename... Rest>
inline constexpr calls_indirectly_key::value_t<First, Rest...> calls_indirectly;
} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {

template <>
struct is_property_key<ext::intel::experimental::indirectly_callable_key>
    : std::true_type {};
template <>
struct is_property_key<ext::intel::experimental::calls_indirectly_key>
    : std::true_type {};

namespace detail {

template <>
struct IsCompileTimeProperty<ext::intel::experimental::indirectly_callable_key>
    : std::true_type {};
template <>
struct IsCompileTimeProperty<ext::intel::experimental::calls_indirectly_key>
    : std::true_type {};

template <>
struct PropertyToKind<ext::intel::experimental::indirectly_callable_key> {
  static constexpr PropKind Kind = PropKind::IndirectlyCallable;
};

template <>
struct PropertyToKind<ext::intel::experimental::calls_indirectly_key> {
  static constexpr PropKind Kind = PropKind::CallsIndirectly;
};

template <typename Set>
struct PropertyMetaInfo<
    ext::intel::experimental::indirectly_callable_key::value_t<Set>> {
  static constexpr const char *name = "indirectly-callable";
  static constexpr const char *value =
#ifdef __SYCL_DEVICE_ONLY__
      __builtin_sycl_unique_stable_name(Set);
#else
      "";
#endif
};

template <typename First, typename... Rest>
struct PropertyMetaInfo<
    ext::intel::experimental::calls_indirectly_key::value_t<First, Rest...>> {
  static constexpr const char *name = "calls-indirectly";
  static constexpr const char *value =
#ifdef __SYCL_DEVICE_ONLY__
      // FIXME: we should handle Rest... here as well
      __builtin_sycl_unique_stable_name(First);
#else
      "";
#endif
};

template <typename T = void> struct void_or_T {
  using type = T;
};

} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

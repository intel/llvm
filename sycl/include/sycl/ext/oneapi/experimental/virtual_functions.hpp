#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
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

template <> struct is_property_key<indirectly_callable_key> : std::true_type {};
template <> struct is_property_key<calls_indirectly_key> : std::true_type {};

namespace detail {

template <>
struct IsCompileTimeProperty<indirectly_callable_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<calls_indirectly_key> : std::true_type {};

template <> struct PropertyToKind<indirectly_callable_key> {
  static constexpr PropKind Kind = PropKind::IndirectlyCallable;
};

template <> struct PropertyToKind<calls_indirectly_key> {
  static constexpr PropKind Kind = PropKind::CallsIndirectly;
};

template <typename Set>
struct PropertyMetaInfo<indirectly_callable_key::value_t<Set>> {
  static constexpr const char *name = "indirectly-callable";
  static constexpr const char *value =
#ifdef __SYCL_DEVICE_ONLY__
      __builtin_sycl_unique_stable_name(Set);
#else
      "";
#endif
};

template <typename First, typename... Rest>
struct PropertyMetaInfo<calls_indirectly_key::value_t<First, Rest...>> {
  static constexpr const char *name = "calls-indirectly";
  static constexpr const char *value =
#ifdef __SYCL_DEVICE_ONLY__
      // FIXME: we should handle Rest... here as well
      __builtin_sycl_unique_stable_name(First);
#else
      "";
#endif
};

} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

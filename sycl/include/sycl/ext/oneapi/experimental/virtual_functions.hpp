#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_utils.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <utility>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
struct indirectly_callable_key
    : detail::compile_time_property_key<detail::PropKind::IndirectlyCallable> {
  template <typename Set>
  using value_t =
      sycl::ext::oneapi::experimental::property_value<indirectly_callable_key,
                                                      Set>;
};

inline constexpr indirectly_callable_key::value_t<void> indirectly_callable;

template <typename Set>
inline constexpr indirectly_callable_key::value_t<Set> indirectly_callable_in;

struct calls_indirectly_key
    : detail::compile_time_property_key<detail::PropKind::CallsIndirectly> {
  template <typename... SetIds>
  using value_t =
      sycl::ext::oneapi::experimental::property_value<calls_indirectly_key,
                                                      SetIds...>;
};

inline constexpr calls_indirectly_key::value_t<void> assume_indirect_calls;

template <typename... SetIds>
inline constexpr calls_indirectly_key::value_t<SetIds...>
    assume_indirect_calls_to;

namespace detail {

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

#ifdef __SYCL_DEVICE_ONLY__
// Helper to concatenate several lists of characters into a single string.
// Lists are separated from each other with comma within the resulting string.
template <typename List, typename... Rest> struct ConcatenateCharsToStr;

// Specialization for a single list
template <char... Chars> struct ConcatenateCharsToStr<CharList<Chars...>> {
  static constexpr char value[] = {Chars..., '\0'};
};

// Specialization for two lists
template <char... Chars, char... CharsToAppend>
struct ConcatenateCharsToStr<CharList<Chars...>, CharList<CharsToAppend...>>
    : ConcatenateCharsToStr<CharList<Chars..., ',', CharsToAppend...>> {};

// Specialization for the case when there are more than two lists
template <char... Chars, char... CharsToAppend, typename... Rest>
struct ConcatenateCharsToStr<CharList<Chars...>, CharList<CharsToAppend...>,
                             Rest...>
    : ConcatenateCharsToStr<CharList<Chars..., ',', CharsToAppend...>,
                            Rest...> {};

// Helper to convert type T to a list of characters representing the type (its
// mangled name).
template <typename T, size_t... Indices> struct StableNameToCharsHelper {
  using chars = CharList<__builtin_sycl_unique_stable_name(T)[Indices]...>;
};

// Wrapper helper for the struct above
template <typename T, typename Sequence> struct StableNameToChars;

// Specialization of that wrapper helper which accepts sequence of integers
template <typename T, size_t... Indices>
struct StableNameToChars<T, std::integer_sequence<size_t, Indices...>>
    : StableNameToCharsHelper<T, Indices...> {};

// Creates a comma-separated string with unique stable names for each type in
// Ts.
template <typename... Ts>
struct UniqueStableNameListStr
    : ConcatenateCharsToStr<typename StableNameToChars<
          Ts, std::make_index_sequence<__builtin_strlen(
                  __builtin_sycl_unique_stable_name(Ts))>>::chars...> {};
#endif // __SYCL_DEVICE_ONLY__

template <typename... SetIds>
struct PropertyMetaInfo<calls_indirectly_key::value_t<SetIds...>> {
  static constexpr const char *name = "calls-indirectly";
  static constexpr const char *value =
#ifdef __SYCL_DEVICE_ONLY__
      UniqueStableNameListStr<SetIds...>::value;
#else
      "";
#endif
};

} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

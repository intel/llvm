//==--------- SYCL annotated_arg/ptr properties for caching control --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

enum class cache_mode {
  uncached,
  cached,
  streaming,
  invalidate,
  constant,
  write_through,
  write_back
};
using cache_mode = sycl::ext::intel::experimental::cache_mode;
using cache_level = sycl::ext::oneapi::experimental::cache_level;

namespace detail {

template <int count> static constexpr void checkLevel1() {
  static_assert(count < 2, "Duplicate cache_level L1 specification");
}
template <int count> static constexpr void checkLevel2() {
  static_assert(count < 2, "Duplicate cache_level L2 specification");
}
template <int count> static constexpr void checkLevel3() {
  static_assert(count < 2, "Duplicate cache_level L3 specification");
}
template <int count> static constexpr void checkLevel4() {
  static_assert(count < 2, "Duplicate cache_level L4 specification");
}

} // namespace detail

template <cache_mode M, cache_level... Ls> struct cache_control {
  static constexpr const auto mode = M;
  static constexpr const int countL1 = ((Ls == cache_level::L1 ? 1 : 0) + ...);
  static constexpr const int countL2 = ((Ls == cache_level::L2 ? 1 : 0) + ...);
  static constexpr const int countL3 = ((Ls == cache_level::L3 ? 1 : 0) + ...);
  static constexpr const int countL4 = ((Ls == cache_level::L4 ? 1 : 0) + ...);
  static constexpr const uint32_t levels = ((1 << static_cast<int>(Ls)) | ...);
  static constexpr const uint32_t encoding =
      (countL1, countL2, countL3, countL4, detail::checkLevel1<countL1>(),
       detail::checkLevel2<countL2>(), detail::checkLevel3<countL3>(),
       detail::checkLevel4<countL4>(), levels << static_cast<int>(M) * 4);
};

template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;

struct read_hint_key {
  template <typename... Cs>
  using value_t = property_value<read_hint_key, Cs...>;
};

struct read_assertion_key {
  template <typename... Cs>
  using value_t = property_value<read_assertion_key, Cs...>;
};

struct write_hint_key {
  template <typename... Cs>
  using value_t = property_value<write_hint_key, Cs...>;
};

template <typename... Cs>
inline constexpr read_hint_key::value_t<Cs...> read_hint;

template <typename... Cs>
inline constexpr read_assertion_key::value_t<Cs...> read_assertion;

template <typename... Cs>
inline constexpr write_hint_key::value_t<Cs...> write_hint;

} // namespace experimental
} // namespace intel

namespace oneapi {
namespace experimental {

template <typename T, typename PropertyListT> class annotated_arg;
template <typename T, typename PropertyListT> class annotated_ptr;

using read_hint_key = intel::experimental::read_hint_key;
template <> struct is_property_key<read_hint_key> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<read_hint_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<read_hint_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

using read_assertion_key = intel::experimental::read_assertion_key;
template <> struct is_property_key<read_assertion_key> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<read_assertion_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<read_assertion_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

using write_hint_key = intel::experimental::write_hint_key;
template <> struct is_property_key<write_hint_key> : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<write_hint_key, annotated_arg<T, PropertyListT>>
    : std::true_type {};
template <typename T, typename PropertyListT>
struct is_property_key_of<write_hint_key, annotated_ptr<T, PropertyListT>>
    : std::true_type {};

namespace detail {

static constexpr int countL(int levels, int mask) {
  return levels & mask ? 1 : 0;
}

template <int countL1, int countL2, int countL3, int countL4>
static constexpr void checkUnique() {
  static_assert(countL1 < 2, "Conflicting cache_mode at L1");
  static_assert(countL2 < 2, "Conflicting cache_mode at L2");
  static_assert(countL3 < 2, "Conflicting cache_mode at L3");
  static_assert(countL4 < 2, "Conflicting cache_mode at L4");
}

using cache_mode = sycl::ext::intel::experimental::cache_mode;

template <cache_mode M> static constexpr int checkReadHint() {
  static_assert(
      M == cache_mode::uncached || M == cache_mode::cached ||
          M == cache_mode::streaming,
      "read_hint must specify cache_mode uncached, cached or streaming");
  return 0;
}

template <cache_mode M> static constexpr int checkReadAssertion() {
  static_assert(
      M == cache_mode::invalidate || M == cache_mode::constant,
      "read_assertion must specify cache_mode invalidate or constant");
  return 0;
}

template <cache_mode M> static constexpr int checkWriteHint() {
  static_assert(M == cache_mode::uncached || M == cache_mode::write_through ||
                    M == cache_mode::write_back || M == cache_mode::streaming,
                "write_hint must specify cache_mode uncached, write_through, "
                "write_back or streaming");
  return 0;
}

template <> struct PropertyToKind<read_hint_key> {
  static constexpr PropKind Kind = PropKind::CacheControlRead;
};
template <> struct IsCompileTimeProperty<read_hint_key> : std::true_type {};
template <typename... Cs>
struct PropertyMetaInfo<read_hint_key::value_t<Cs...>> {
  static constexpr const char *name = "sycl-cache-read";
  static constexpr const int value =
      ((checkReadHint<Cs::mode>() + ...),
       checkUnique<(countL(Cs::levels, 1) + ...), (countL(Cs::levels, 2) + ...),
                   (countL(Cs::levels, 4) + ...),
                   (countL(Cs::levels, 8) + ...)>(),
       ((Cs::encoding) | ...));
};

template <> struct PropertyToKind<read_assertion_key> {
  static constexpr PropKind Kind = PropKind::CacheControlRead;
};
template <>
struct IsCompileTimeProperty<read_assertion_key> : std::true_type {};
template <typename... Cs>
struct PropertyMetaInfo<read_assertion_key::value_t<Cs...>> {
  static constexpr const char *name = "sycl-cache-read";
  static constexpr const int value =
      ((checkReadAssertion<Cs::mode>() + ...),
       checkUnique<(countL(Cs::levels, 1) + ...), (countL(Cs::levels, 2) + ...),
                   (countL(Cs::levels, 4) + ...),
                   (countL(Cs::levels, 8) + ...)>(),
       ((Cs::encoding) | ...));
};

template <> struct PropertyToKind<write_hint_key> {
  static constexpr PropKind Kind = PropKind::CacheControlWrite;
};
template <> struct IsCompileTimeProperty<write_hint_key> : std::true_type {};
template <typename... Cs>
struct PropertyMetaInfo<write_hint_key::value_t<Cs...>> {
  static constexpr const char *name = "sycl-cache-write";
  static constexpr const int value =
      ((checkWriteHint<Cs::mode>() + ...),
       checkUnique<(countL(Cs::levels, 1) + ...), (countL(Cs::levels, 2) + ...),
                   (countL(Cs::levels, 4) + ...),
                   (countL(Cs::levels, 8) + ...)>(),
       ((Cs::encoding) | ...));
};

} // namespace detail

template <typename T, typename... Cs>
struct is_valid_property<T, read_hint_key::value_t<Cs...>>
    : std::bool_constant<std::is_pointer<T>::value> {};

template <typename T, typename... Cs>
struct is_valid_property<T, read_assertion_key::value_t<Cs...>>
    : std::bool_constant<std::is_pointer<T>::value> {};

template <typename T, typename... Cs>
struct is_valid_property<T, write_hint_key::value_t<Cs...>>
    : std::bool_constant<std::is_pointer<T>::value> {};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

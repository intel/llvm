//==--------- SYCL annotated_arg/ptr properties for caching control --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/common_annotated_properties/properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

enum class level : std::uint16_t { L1=1, L2, L3, L4 };

template <typename PropertyT, typename... Ts>
using property_value =
    sycl::ext::oneapi::experimental::property_value<PropertyT, Ts...>;

#define __SYCL_CACHE_CONTROL_M1(P)                                             \
  struct P {                                                                   \
    template <level... Ls>                                                     \
    using value_t = property_value<P, std::integral_constant<level, Ls>...>;   \
  };

__SYCL_CACHE_CONTROL_M1(cache_control_read_cached_key)
__SYCL_CACHE_CONTROL_M1(cache_control_read_uncached_key)
__SYCL_CACHE_CONTROL_M1(cache_control_read_streaming_key)
__SYCL_CACHE_CONTROL_M1(cache_control_invalidate_after_read_key)
__SYCL_CACHE_CONTROL_M1(cache_control_read_const_cached_key)
__SYCL_CACHE_CONTROL_M1(cache_control_write_uncached_key)
__SYCL_CACHE_CONTROL_M1(cache_control_write_streaming_key)
__SYCL_CACHE_CONTROL_M1(cache_control_write_through_key)
__SYCL_CACHE_CONTROL_M1(cache_control_write_back_key)

} // namespace experimental
} // namespace intel

namespace oneapi {
namespace experimental {

template <typename T, typename PropertyListT> class annotated_arg;
template <typename T, typename PropertyListT> class annotated_ptr;

#define __SYCL_CACHE_CONTROL_M2(P)                                             \
  using P = intel::experimental::P;                                            \
  template <> struct is_property_key<P> : std::true_type {};                   \
  template <typename T, typename PropertyListT>                                \
  struct is_property_key_of<P, annotated_arg<T, PropertyListT>>                \
      : std::true_type {};                                                     \
  template <typename T, typename PropertyListT>                                \
  struct is_property_key_of<P, annotated_ptr<T, PropertyListT>>                \
      : std::true_type {};

__SYCL_CACHE_CONTROL_M2(cache_control_read_cached_key)
__SYCL_CACHE_CONTROL_M2(cache_control_read_uncached_key)
__SYCL_CACHE_CONTROL_M2(cache_control_read_streaming_key)
__SYCL_CACHE_CONTROL_M2(cache_control_invalidate_after_read_key)
__SYCL_CACHE_CONTROL_M2(cache_control_read_const_cached_key)
__SYCL_CACHE_CONTROL_M2(cache_control_write_uncached_key)
__SYCL_CACHE_CONTROL_M2(cache_control_write_streaming_key)
__SYCL_CACHE_CONTROL_M2(cache_control_write_through_key)
__SYCL_CACHE_CONTROL_M2(cache_control_write_back_key)

using namespace intel::experimental;

#define __SYCL_CACHE_CONTROL_M3(P)                                             \
  template <level... Ls> inline constexpr P##_key::value_t<Ls...> P;

__SYCL_CACHE_CONTROL_M3(cache_control_read_cached)
__SYCL_CACHE_CONTROL_M3(cache_control_read_uncached)
__SYCL_CACHE_CONTROL_M3(cache_control_read_streaming)
__SYCL_CACHE_CONTROL_M3(cache_control_invalidate_after_read)
__SYCL_CACHE_CONTROL_M3(cache_control_read_const_cached)
__SYCL_CACHE_CONTROL_M3(cache_control_write_uncached)
__SYCL_CACHE_CONTROL_M3(cache_control_write_streaming)
__SYCL_CACHE_CONTROL_M3(cache_control_write_through)
__SYCL_CACHE_CONTROL_M3(cache_control_write_back)

namespace detail {

template <level L_1> static constexpr void checkLevels() {}
template <level L_1, level L_2> static constexpr void checkLevels() {
  static_assert(L_1 != L_2, "Duplicate cache level specification.");
}
template <level L_1, level L_2, level L_3> static constexpr void checkLevels() {
  static_assert(L_1 != L_2 && L_1 != L_3 && L_2 != L_3,
                "Duplicate cache level specification.");
}
template <level L_1, level L_2, level L_3, level L_4>
static constexpr void checkLevels() {
  static_assert(L_1 != L_2 && L_1 != L_3 && L_1 != L_4 && L_2 != L_3 &&
                    L_2 != L_4 && L_3 != L_4,
                "Duplicate cache level specification.");
}

#define __SYCL_CACHE_CONTROL_M4(P, K, N)                                       \
  template <> struct PropertyToKind<P> {                                       \
    static constexpr PropKind Kind = PropKind::K;                              \
  };                                                                           \
  template <> struct IsCompileTimeProperty<P> : std::true_type {};             \
  template <level... Ls> struct PropertyMetaInfo<P::value_t<Ls...>> {          \
    static constexpr const char *name = N;                                     \
    static constexpr const int value =                                         \
        (checkLevels<Ls...>(), ((1 << (static_cast<int>(Ls) - 1)) | ...));     \
  };

__SYCL_CACHE_CONTROL_M4(cache_control_read_cached_key, CacheControlReadCached,
                        "sycl-cache-read-cached")
__SYCL_CACHE_CONTROL_M4(cache_control_read_uncached_key,
                        CacheControlReadUncached, "sycl-cache-read-uncached")
__SYCL_CACHE_CONTROL_M4(cache_control_read_streaming_key,
                        CacheControlReadStreaming, "sycl-cache-read-streaming")
__SYCL_CACHE_CONTROL_M4(cache_control_invalidate_after_read_key,
                        CacheControlReadInvalidateAfterRead,
                        "sycl-cache-read-invalidate-after-read")
__SYCL_CACHE_CONTROL_M4(cache_control_read_const_cached_key,
                        CacheControlReadConstCached,
                        "sycl-cache-read-const-cached")
__SYCL_CACHE_CONTROL_M4(cache_control_write_uncached_key,
                        CacheControlWriteUncached, "sycl-cache-write-uncached")
__SYCL_CACHE_CONTROL_M4(cache_control_write_streaming_key,
                        CacheControlWriteStreaming,
                        "sycl-cache-write-streaming")
__SYCL_CACHE_CONTROL_M4(cache_control_write_through_key,
                        CacheControlWriteThrough, "sycl-cache-write-through")
__SYCL_CACHE_CONTROL_M4(cache_control_write_back_key, CacheControlWriteBack,
                        "sycl-cache-write-back")

} // namespace detail

#define __SYCL_CACHE_CONTROL_M5(P)                                             \
  template <typename T, level... Ts>                                           \
  struct is_valid_property<T, P::value_t<Ts...>>                               \
      : std::bool_constant<std::is_pointer<T>::value> {};

__SYCL_CACHE_CONTROL_M5(cache_control_read_cached_key)
__SYCL_CACHE_CONTROL_M5(cache_control_read_uncached_key)
__SYCL_CACHE_CONTROL_M5(cache_control_read_streaming_key)
__SYCL_CACHE_CONTROL_M5(cache_control_invalidate_after_read_key)
__SYCL_CACHE_CONTROL_M5(cache_control_read_const_cached_key)
__SYCL_CACHE_CONTROL_M5(cache_control_write_uncached_key)
__SYCL_CACHE_CONTROL_M5(cache_control_write_streaming_key)
__SYCL_CACHE_CONTROL_M5(cache_control_write_through_key)
__SYCL_CACHE_CONTROL_M5(cache_control_write_back_key)

#undef __SYCL_CACHE_CONTROL_M1
#undef __SYCL_CACHE_CONTROL_M2
#undef __SYCL_CACHE_CONTROL_M3
#undef __SYCL_CACHE_CONTROL_M4
#undef __SYCL_CACHE_CONTROL_M5

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

//==---- kernel_properties.hpp --- sycl_khr_kernel_properties extension ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the sycl_khr_kernel_properties extension: the
// `khr::property::tag::kernel` tag type, the `work_group_size`,
// `work_group_size_hint` and `sub_group_size` kernel properties, and the
// machinery that extracts properties embedded in a named function object via
// a `get(khr::property::tag::kernel) const` member function.
//
// The embedded properties reuse the compile-time property infrastructure of
// sycl_ext_oneapi_kernel_properties: each khr property is lowered to the same
// IR attribute as its oneapi counterpart (see `PropertyMetaInfo` below), so the
// existing device lowering and kernel information queries apply unchanged.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#include <sycl/ext/oneapi/properties.hpp>
#include <sycl/ext/oneapi/properties/property_utils.hpp>
#include <sycl/feature_test.hpp> // for SYCL_KHR_KERNEL_PROPERTIES
#include <sycl/khr/properties.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace khr {
namespace property {

namespace tag {
// Unlike the other tag types, this one is complete: it is also the parameter
// type of the `get` member function used to embed kernel properties.
struct kernel {};
} // namespace tag

namespace key {
struct work_group_size
    : property::detail::property_key_base<work_group_size, true> {};
struct work_group_size_hint
    : property::detail::property_key_base<work_group_size_hint, true> {};
struct sub_group_size
    : property::detail::property_key_base<sub_group_size, true> {};
} // namespace key

namespace detail {
template <std::size_t... Dims> constexpr bool all_non_zero() {
  return ((Dims != 0) && ...);
}

template <std::size_t... Dims>
struct work_group_size_property
    : property_base<work_group_size_property<Dims...>, key::work_group_size> {
  static_assert(sizeof...(Dims) >= 1 && sizeof...(Dims) <= 3,
                "sycl::khr::property::work_group_size requires 1, 2 or 3 "
                "dimensions.");
  static_assert(all_non_zero<Dims...>(),
                "sycl::khr::property::work_group_size must only contain "
                "non-zero values.");
  static constexpr std::size_t value[sizeof...(Dims)] = {Dims...};
};

template <std::size_t... Dims>
struct work_group_size_hint_property
    : property_base<work_group_size_hint_property<Dims...>,
                    key::work_group_size_hint> {
  static_assert(sizeof...(Dims) >= 1 && sizeof...(Dims) <= 3,
                "sycl::khr::property::work_group_size_hint requires 1, 2 or 3 "
                "dimensions.");
  static_assert(all_non_zero<Dims...>(),
                "sycl::khr::property::work_group_size_hint must only contain "
                "non-zero values.");
  static constexpr std::size_t value[sizeof...(Dims)] = {Dims...};
};

template <std::uint32_t Size>
struct sub_group_size_property
    : property_base<sub_group_size_property<Size>, key::sub_group_size> {
  static_assert(Size != 0, "sycl::khr::property::sub_group_size must be a "
                           "non-zero value.");
  static constexpr std::uint32_t value = Size;
};
} // namespace detail

template <std::size_t... Dims>
inline constexpr detail::work_group_size_property<Dims...> work_group_size;

template <std::size_t... Dims>
inline constexpr detail::work_group_size_hint_property<Dims...>
    work_group_size_hint;

template <std::uint32_t Size>
inline constexpr detail::sub_group_size_property<Size> sub_group_size;

} // namespace property

template <>
struct is_property_key_for<property::key::work_group_size,
                           property::tag::kernel> : std::true_type {};
template <>
struct is_property_key_for<property::key::work_group_size_hint,
                           property::tag::kernel> : std::true_type {};
template <>
struct is_property_key_for<property::key::sub_group_size, property::tag::kernel>
    : std::true_type {};

namespace property::detail {

// Maps each khr kernel property to the equivalent
// sycl_ext_oneapi_kernel_properties property. Used where the kernel properties
// have to be handed over to code paths that only understand oneapi property
// lists (e.g. reductions, where the user's function object is wrapped).
template <typename Property> struct to_oneapi_kernel_property;
template <std::size_t... Dims>
struct to_oneapi_kernel_property<work_group_size_property<Dims...>> {
  using type = ext::oneapi::experimental::work_group_size_key::value_t<Dims...>;
};
template <std::size_t... Dims>
struct to_oneapi_kernel_property<work_group_size_hint_property<Dims...>> {
  using type =
      ext::oneapi::experimental::work_group_size_hint_key::value_t<Dims...>;
};
template <std::uint32_t Size>
struct to_oneapi_kernel_property<sub_group_size_property<Size>> {
  using type = ext::oneapi::experimental::sub_group_size_key::value_t<Size>;
};

// Return type of `declval<T>().get(tag::kernel{})`, if well formed.
template <typename T, typename = void>
struct has_kernel_properties_get : std::false_type {};
template <typename T>
struct has_kernel_properties_get<T, std::void_t<decltype(std::declval<T>().get(
                                        std::declval<tag::kernel>()))>>
    : std::true_type {
  using return_type =
      decltype(std::declval<T>().get(std::declval<tag::kernel>()));
};

// Unpacks a property or property list into `Pack<Properties...>`.
template <template <typename...> class Pack, typename R, typename = void>
struct unpack_kernel_properties {
  using type = Pack<>;
};
template <template <typename...> class Pack, typename... Ps>
struct unpack_kernel_properties<Pack, properties<Ps...>> {
  using type = Pack<Ps...>;
};
template <template <typename...> class Pack, typename R>
struct unpack_kernel_properties<Pack, R, std::enable_if_t<is_property_v<R>>> {
  using type = Pack<R>;
};

template <typename... Ps> struct oneapi_kernel_properties_of {
  using type = decltype(ext::oneapi::experimental::properties{
      typename to_oneapi_kernel_property<Ps>::type{}...});
};

// The kernel properties embedded in the (decayed) kernel function object type
// `KernelType`.
template <typename KernelType> struct embedded_kernel_properties_impl {
private:
  using const_get = has_kernel_properties_get<const KernelType &>;

  template <typename T, bool = T::value> struct return_type_of {
    using type = properties<>;
  };
  template <typename T> struct return_type_of<T, true> {
    using type = typename T::return_type;
  };

public:
  static constexpr bool has_non_const_get =
      !const_get::value && (has_kernel_properties_get<KernelType>::value ||
                            has_kernel_properties_get<KernelType &>::value);

  using return_type = typename return_type_of<const_get>::type;
  static constexpr bool is_valid =
      is_property_or_list_for_v<return_type, tag::kernel>;

  // `Pack<Properties...>` with the embedded properties. Empty if there is no
  // (valid) getter, so that only the diagnostics below are reported.
  template <template <typename...> class Pack>
  using unpack_t = typename unpack_kernel_properties<
      Pack, std::conditional_t<is_valid, return_type, properties<>>>::type;

  // The equivalent sycl_ext_oneapi_kernel_properties property list.
  using oneapi_properties_t =
      typename unpack_t<oneapi_kernel_properties_of>::type;
};

template <typename KernelType>
using embedded_kernel_properties = embedded_kernel_properties_impl<
    std::remove_cv_t<std::remove_reference_t<KernelType>>>;

// Library-side diagnostics of the extension. Conflicts between the embedded
// properties and kernel attributes, or properties embedded via
// get(sycl::ext::oneapi::experimental::properties_tag), are diagnosed by the
// device compiler.
template <typename KernelType> constexpr void diagnose_kernel_properties_get() {
  using Props = embedded_kernel_properties_impl<KernelType>;
  static_assert(!Props::has_non_const_get,
                "get(sycl::khr::property::tag::kernel) member in kernel "
                "function object must be declared as a const member function.");
  static_assert(Props::is_valid,
                "The return type of get(sycl::khr::property::tag::kernel) must "
                "be a property or a property list for which "
                "sycl::khr::is_property_or_list_for_v<R, "
                "sycl::khr::property::tag::kernel> is true.");
}

template <typename KernelType>
constexpr void diagnose_embedded_kernel_properties() {
  diagnose_kernel_properties_get<
      std::remove_cv_t<std::remove_reference_t<KernelType>>>();
}

} // namespace property::detail
} // namespace khr

namespace ext::oneapi::experimental::detail {
// The khr kernel properties lower to the same IR attributes as their
// sycl_ext_oneapi_kernel_properties counterparts.
template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<
    khr::property::detail::work_group_size_property<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value = SizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<
    khr::property::detail::work_group_size_hint_property<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value = SizeListToStr<Dim0, Dims...>::value;
};
template <uint32_t Size>
struct PropertyMetaInfo<khr::property::detail::sub_group_size_property<Size>> {
  static constexpr const char *name = "sycl-sub-group-size";
  static constexpr uint32_t value = Size;
};
} // namespace ext::oneapi::experimental::detail

} // namespace _V1
} // namespace sycl

#endif // __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

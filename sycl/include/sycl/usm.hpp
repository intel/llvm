//==---------------- usm.hpp - SYCL USM ------------------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/khr/properties.hpp>
#include <sycl/platform.hpp>
#include <sycl/property_list.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm/usm_pointer_info.hpp>

#include <cstddef>   // for size_t

namespace sycl {
inline namespace _V1 {

class context;

///
// Explicit USM
///
__SYCL_EXPORT void *malloc_device(
    size_t size, const device &dev, const context &ctxt,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_device(
    size_t size, const device &dev, const context &ctxt,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_device(
    size_t size, const queue &q,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_device(
    size_t size, const queue &q, const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

__SYCL_EXPORT void *aligned_alloc_device(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_device(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_device(
    size_t alignment, size_t size, const queue &q,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_device(
    size_t alignment, size_t size, const queue &q,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

__SYCL_EXPORT void
free(void *ptr, const context &ctxt,
     const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void
free(void *ptr, const queue &q,
     const detail::code_location &CodeLoc = detail::code_location::current());

///
// Restricted USM
///
__SYCL_EXPORT void *malloc_host(
    size_t size, const context &ctxt,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_host(
    size_t size, const context &ctxt, const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_host(
    size_t size, const queue &q,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_host(
    size_t size, const queue &q, const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

__SYCL_EXPORT void *malloc_shared(
    size_t size, const device &dev, const context &ctxt,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_shared(
    size_t size, const device &dev, const context &ctxt,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_shared(
    size_t size, const queue &q,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *malloc_shared(
    size_t size, const queue &q, const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

__SYCL_EXPORT void *aligned_alloc_host(
    size_t alignment, size_t size, const context &ctxt,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_host(
    size_t alignment, size_t size, const context &ctxt,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_host(
    size_t alignment, size_t size, const queue &q,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_host(
    size_t alignment, size_t size, const queue &q,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

__SYCL_EXPORT void *aligned_alloc_shared(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_shared(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_shared(
    size_t alignment, size_t size, const queue &q,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc_shared(
    size_t alignment, size_t size, const queue &q,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

///
// single form
///
__SYCL_EXPORT void *
malloc(size_t size, const device &dev, const context &ctxt, usm::alloc kind,
       const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *
malloc(size_t size, const device &dev, const context &ctxt, usm::alloc kind,
       const property_list &propList,
       const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *
malloc(size_t size, const queue &q, usm::alloc kind,
       const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *
malloc(size_t size, const queue &q, usm::alloc kind,
       const property_list &propList,
       const detail::code_location &CodeLoc = detail::code_location::current());

__SYCL_EXPORT void *aligned_alloc(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    usm::alloc kind,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    usm::alloc kind, const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc(
    size_t alignment, size_t size, const queue &q, usm::alloc kind,
    const detail::code_location &CodeLoc = detail::code_location::current());
__SYCL_EXPORT void *aligned_alloc(
    size_t alignment, size_t size, const queue &q, usm::alloc kind,
    const property_list &propList,
    const detail::code_location &CodeLoc = detail::code_location::current());

///
// Helper function used to determine if the Alignment argument is a power of 2
///
inline size_t is_not_power_of_two(size_t Alignment) {
  return (Alignment & (Alignment - 1));
}

///
// Template forms
///
template <typename T>
T *malloc_device(
    size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return static_cast<T *>(aligned_alloc_device(alignof(T), Count * sizeof(T),
                                               Dev, Ctxt, PropList, CodeLoc));
}

template <typename T>
T *malloc_device(
    size_t Count, const queue &Q, const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_device<T>(Count, Q.get_device(), Q.get_context(), PropList,
                          CodeLoc);
}

template <typename T>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  size_t Align = Alignment < alignof(T) ? alignof(T) : Alignment;
  return static_cast<T *>(aligned_alloc_device(Align, Count * sizeof(T), Dev,
                                               Ctxt, PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  return aligned_alloc_device<T>(Alignment, Count, Q.get_device(),
                                 Q.get_context(), PropList, CodeLoc);
}

template <typename T>
T *malloc_host(
    size_t Count, const context &Ctxt, const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return static_cast<T *>(aligned_alloc_host(alignof(T), Count * sizeof(T),
                                             Ctxt, PropList, CodeLoc));
}

template <typename T>
T *malloc_host(
    size_t Count, const queue &Q, const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_host<T>(Count, Q.get_context(), PropList, CodeLoc);
}

template <typename T>
T *malloc_shared(
    size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return static_cast<T *>(aligned_alloc_shared(alignof(T), Count * sizeof(T),
                                               Dev, Ctxt, PropList, CodeLoc));
}

template <typename T>
T *malloc_shared(
    size_t Count, const queue &Q, const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_shared<T>(Count, Q.get_device(), Q.get_context(), PropList,
                          CodeLoc);
}

template <typename T>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const context &Ctxt,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  size_t Align = Alignment < alignof(T) ? alignof(T) : Alignment;
  return static_cast<T *>(
      aligned_alloc_host(Align, Count * sizeof(T), Ctxt, PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  return aligned_alloc_host<T>(Alignment, Count, Q.get_context(), PropList,
                               CodeLoc);
}

template <typename T>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  size_t Align = Alignment < alignof(T) ? alignof(T) : Alignment;
  return static_cast<T *>(aligned_alloc_shared(Align, Count * sizeof(T), Dev,
                                               Ctxt, PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  return aligned_alloc_shared<T>(Alignment, Count, Q.get_device(),
                                 Q.get_context(), PropList, CodeLoc);
}

template <typename T>
T *malloc(
    size_t Count, const device &Dev, const context &Ctxt, usm::alloc Kind,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return static_cast<T *>(aligned_alloc(alignof(T), Count * sizeof(T), Dev,
                                        Ctxt, Kind, PropList, CodeLoc));
}

template <typename T>
T *malloc(
    size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc<T>(Count, Q.get_device(), Q.get_context(), Kind, PropList,
                   CodeLoc);
}

template <typename T>
T *aligned_alloc(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    usm::alloc Kind, const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  size_t Align = Alignment < alignof(T) ? alignof(T) : Alignment;
  return static_cast<T *>(aligned_alloc(Align, Count * sizeof(T), Dev, Ctxt,
                                        Kind, PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc(
    size_t Alignment, size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (is_not_power_of_two(Alignment)) {
    return nullptr;
  }
  return aligned_alloc<T>(Alignment, Count, Q.get_device(), Q.get_context(),
                          Kind, PropList, CodeLoc);
}

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
namespace khr::property::tag {
struct usm_device; // tag type, never defined
struct usm_host;   // tag type, never defined
struct usm_shared; // tag type, never defined
} // namespace khr::property::tag

namespace detail {
template <typename PropertyOrList>
inline constexpr bool KhrPropsForUsmDevice =
    khr::property::detail::is_property_or_list_for_v<
        PropertyOrList, khr::property::tag::usm_device>;

template <typename PropertyOrList>
inline constexpr bool KhrPropsForUsmHost =
    khr::property::detail::is_property_or_list_for_v<
        PropertyOrList, khr::property::tag::usm_host>;

template <typename PropertyOrList>
inline constexpr bool KhrPropsForUsmShared =
    khr::property::detail::is_property_or_list_for_v<
        PropertyOrList, khr::property::tag::usm_shared>;

template <typename PropertyOrList>
inline constexpr bool KhrPropsForUsmAny =
    KhrPropsForUsmDevice<PropertyOrList> ||
    KhrPropsForUsmHost<PropertyOrList> || KhrPropsForUsmShared<PropertyOrList>;

template <typename PropertyOrList, usm::alloc Kind>
inline constexpr bool KhrPropsForUsmKind =
    (Kind == usm::alloc::device && KhrPropsForUsmDevice<PropertyOrList>) ||
    (Kind == usm::alloc::host && KhrPropsForUsmHost<PropertyOrList>) ||
    (Kind == usm::alloc::shared && KhrPropsForUsmShared<PropertyOrList>);

// Throws errc::invalid if PropertyOrList is not valid for the runtime Kind.
template <typename PropertyOrList>
void checkKhrPropsForUsmKind(usm::alloc Kind) {
  if constexpr (!(KhrPropsForUsmDevice<PropertyOrList> &&
                  KhrPropsForUsmHost<PropertyOrList> &&
                  KhrPropsForUsmShared<PropertyOrList>)) {
    if ((Kind == usm::alloc::device && !KhrPropsForUsmDevice<PropertyOrList>) ||
        (Kind == usm::alloc::host && !KhrPropsForUsmHost<PropertyOrList>) ||
        (Kind == usm::alloc::shared && !KhrPropsForUsmShared<PropertyOrList>))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Properties are not valid for the USM allocation "
                            "kind");
  }
}
} // namespace detail

// sycl_khr_properties device allocation overloads.
template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
void *malloc_device(
    size_t NumBytes, const device &Dev, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_device(NumBytes, Dev, Ctxt, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
T *malloc_device(
    size_t Count, const device &Dev, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_device<T>(Count, Dev, Ctxt, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
void *malloc_device(
    size_t NumBytes, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_device(NumBytes, Q, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
T *malloc_device(
    size_t Count, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_device<T>(Count, Q, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
void *aligned_alloc_device(
    size_t Alignment, size_t NumBytes, const device &Dev, const context &Ctxt,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_device(Alignment, NumBytes, Dev, Ctxt, property_list{},
                              CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_device<T>(Alignment, Count, Dev, Ctxt, property_list{},
                                 CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
void *aligned_alloc_device(
    size_t Alignment, size_t NumBytes, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_device(Alignment, NumBytes, Q, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmDevice<PropertyOrList>>>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_device<T>(Alignment, Count, Q, property_list{}, CodeLoc);
}

// sycl_khr_properties host allocation overloads.
template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
void *malloc_host(
    size_t NumBytes, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_host(NumBytes, Ctxt, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
T *malloc_host(
    size_t Count, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_host<T>(Count, Ctxt, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
void *malloc_host(
    size_t NumBytes, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_host(NumBytes, Q, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
T *malloc_host(
    size_t Count, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_host<T>(Count, Q, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
void *aligned_alloc_host(
    size_t Alignment, size_t NumBytes, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_host(Alignment, NumBytes, Ctxt, property_list{},
                            CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_host<T>(Alignment, Count, Ctxt, property_list{},
                               CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
void *aligned_alloc_host(
    size_t Alignment, size_t NumBytes, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_host(Alignment, NumBytes, Q, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmHost<PropertyOrList>>>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_host<T>(Alignment, Count, Q, property_list{}, CodeLoc);
}

// sycl_khr_properties shared allocation overloads.
template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
void *malloc_shared(
    size_t NumBytes, const device &Dev, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_shared(NumBytes, Dev, Ctxt, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
T *malloc_shared(
    size_t Count, const device &Dev, const context &Ctxt, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_shared<T>(Count, Dev, Ctxt, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
void *malloc_shared(
    size_t NumBytes, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_shared(NumBytes, Q, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
T *malloc_shared(
    size_t Count, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return malloc_shared<T>(Count, Q, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
void *aligned_alloc_shared(
    size_t Alignment, size_t NumBytes, const device &Dev, const context &Ctxt,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_shared(Alignment, NumBytes, Dev, Ctxt, property_list{},
                              CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_shared<T>(Alignment, Count, Dev, Ctxt, property_list{},
                                 CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
void *aligned_alloc_shared(
    size_t Alignment, size_t NumBytes, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_shared(Alignment, NumBytes, Q, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmShared<PropertyOrList>>>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const queue &Q, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  return aligned_alloc_shared<T>(Alignment, Count, Q, property_list{}, CodeLoc);
}

// sycl_khr_properties parameterized allocation overloads.
template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
void *malloc(
    size_t NumBytes, const device &Dev, const context &Ctxt, usm::alloc Kind,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return malloc(NumBytes, Dev, Ctxt, Kind, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
T *malloc(
    size_t Count, const device &Dev, const context &Ctxt, usm::alloc Kind,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return malloc<T>(Count, Dev, Ctxt, Kind, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
void *malloc(
    size_t NumBytes, const queue &Q, usm::alloc Kind, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return malloc(NumBytes, Q, Kind, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
T *malloc(
    size_t Count, const queue &Q, usm::alloc Kind, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return malloc<T>(Count, Q, Kind, property_list{}, CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
void *aligned_alloc(
    size_t Alignment, size_t NumBytes, const device &Dev, const context &Ctxt,
    usm::alloc Kind, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return aligned_alloc(Alignment, NumBytes, Dev, Ctxt, Kind, property_list{},
                       CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
T *aligned_alloc(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    usm::alloc Kind, PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return aligned_alloc<T>(Alignment, Count, Dev, Ctxt, Kind, property_list{},
                          CodeLoc);
}

template <
    typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
void *aligned_alloc(
    size_t Alignment, size_t NumBytes, const queue &Q, usm::alloc Kind,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return aligned_alloc(Alignment, NumBytes, Q, Kind, property_list{}, CodeLoc);
}

template <
    typename T, typename PropertyOrList,
    typename = std::enable_if_t<detail::KhrPropsForUsmAny<PropertyOrList>>>
T *aligned_alloc(
    size_t Alignment, size_t Count, const queue &Q, usm::alloc Kind,
    PropertyOrList,
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  detail::checkKhrPropsForUsmKind<PropertyOrList>(Kind);
  return aligned_alloc<T>(Alignment, Count, Q, Kind, property_list{}, CodeLoc);
}
#endif // __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

// Device copy enhancement APIs, prepare_for and release_from USM.

namespace ext::oneapi::experimental {

__SYCL_EXPORT void prepare_for_device_copy(const void *Ptr, size_t Size,
                                           const context &Context);

__SYCL_EXPORT void prepare_for_device_copy(const void *Ptr, size_t Size,
                                           const queue &Queue);

__SYCL_EXPORT void release_from_device_copy(const void *Ptr,
                                            const context &Context);

__SYCL_EXPORT void release_from_device_copy(const void *Ptr,
                                            const queue &Queue);

__SYCL_EXPORT void *malloc_device(size_t numBytes, const device &syclDevice,
                                  const property_list &propList = {});

template <typename T>
T *malloc_device(size_t count, const device &syclDevice,
                 const property_list &propList = {}) {
  sycl::context ctxt = syclDevice.get_platform().khr_get_default_context();
  return sycl::malloc_device<T>(count, syclDevice, ctxt, propList);
}

__SYCL_EXPORT void *aligned_alloc_device(size_t alignment, size_t numBytes,
                                         const device &syclDevice,
                                         const property_list &propList = {});

template <typename T>
T *aligned_alloc_device(size_t alignment, size_t count,
                        const device &syclDevice,
                        const property_list &propList = {}) {
  sycl::context ctxt = syclDevice.get_platform().khr_get_default_context();
  return sycl::aligned_alloc_device<T>(alignment, count, syclDevice, ctxt,
                                       propList);
}

__SYCL_EXPORT void *malloc_shared(size_t numBytes, const device &syclDevice,
                                  const property_list &propList = {});

template <typename T>
T *malloc_shared(size_t count, const device &syclDevice,
                 const property_list &propList = {}) {
  sycl::context ctxt = syclDevice.get_platform().khr_get_default_context();
  return sycl::malloc_shared<T>(count, syclDevice, ctxt, propList);
}

__SYCL_EXPORT void *aligned_alloc_shared(size_t alignment, size_t numBytes,
                                         const device &syclDevice,
                                         const property_list &propList = {});

template <typename T>
T *aligned_alloc_shared(size_t alignment, size_t count,
                        const device &syclDevice,
                        const property_list &propList = {}) {
  sycl::context ctxt = syclDevice.get_platform().khr_get_default_context();
  return sycl::aligned_alloc_shared<T>(alignment, count, syclDevice, ctxt,
                                       propList);
}

__SYCL_EXPORT void *malloc(size_t numBytes, const device &syclDevice,
                           usm::alloc kind, const property_list &propList = {});

template <typename T>
T *malloc(size_t count, const device &syclDevice, usm::alloc kind,
          const property_list &propList = {}) {
  sycl::context ctxt = syclDevice.get_platform().khr_get_default_context();
  return sycl::malloc<T>(count, syclDevice, ctxt, kind, propList);
}

__SYCL_EXPORT void *aligned_alloc(size_t alignment, size_t numBytes,
                                  const device &syclDevice, usm::alloc kind,
                                  const property_list &propList = {});

template <typename T>
T *aligned_alloc(size_t alignment, size_t count, const device &syclDevice,
                 usm::alloc kind, const property_list &propList = {}) {
  sycl::context ctxt = syclDevice.get_platform().khr_get_default_context();
  return sycl::aligned_alloc<T>(alignment, count, syclDevice, ctxt, kind,
                                propList);
}
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl

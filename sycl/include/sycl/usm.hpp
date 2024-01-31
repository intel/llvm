//==---------------- usm.hpp - SYCL USM ------------------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/builtins.hpp>      // for max
#include <sycl/context.hpp>       // for context
#include <sycl/detail/common.hpp> // for code_location
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/device.hpp>        // for device
#include <sycl/property_list.hpp> // for property_list
#include <sycl/queue.hpp>         // for queue
#include <sycl/usm/usm_enums.hpp> // for alloc

#include <algorithm> // for max
#include <cstddef>   // for size_t

namespace sycl {
inline namespace _V1 {
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
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return static_cast<T *>(aligned_alloc_device(max(Alignment, alignof(T)),
                                                 Count * sizeof(T), Dev, Ctxt,
                                                 PropList, CodeLoc));
  } else {
    return nullptr;
  }
}

template <typename T>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return aligned_alloc_device<T>(Alignment, Count, Q.get_device(),
                                   Q.get_context(), PropList, CodeLoc);
  } else {
    return nullptr;
  }
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
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return static_cast<T *>(aligned_alloc_host(std ::max(Alignment, alignof(T)),
                                               Count * sizeof(T), Ctxt,
                                               PropList, CodeLoc));
  } else {
    return nullptr;
  }
}

template <typename T>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return aligned_alloc_host<T>(Alignment, Count, Q.get_context(), PropList,
                                 CodeLoc);
  } else {
    return nullptr;
  }
}

template <typename T>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return static_cast<T *>(aligned_alloc_shared(max(Alignment, alignof(T)),
                                                 Count * sizeof(T), Dev, Ctxt,
                                                 PropList, CodeLoc));
  } else {
    return nullptr;
  }
}

template <typename T>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return aligned_alloc_shared<T>(Alignment, Count, Q.get_device(),
                                   Q.get_context(), PropList, CodeLoc);
  } else {
    return nullptr;
  }
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
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return static_cast<T *>(aligned_alloc(max(Alignment, alignof(T)),
                                          Count * sizeof(T), Dev, Ctxt, Kind,
                                          PropList, CodeLoc));
  } else {
    return nullptr;
  }
}

template <typename T>
T *aligned_alloc(
    size_t Alignment, size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {},
    const detail::code_location &CodeLoc = detail::code_location::current()) {
  if (Alignment && !(Alignment & (Alignment - 1))) {
    return aligned_alloc<T>(Alignment, Count, Q.get_device(), Q.get_context(),
                            Kind, PropList, CodeLoc);
  } else {
    return nullptr;
  }
}

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

} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl

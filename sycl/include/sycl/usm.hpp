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
#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/usm/usm_allocator.hpp>
#include <sycl/usm/usm_enums.hpp>

#include <cstddef>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
///
// Explicit USM
///
__SYCL_EXPORT void *malloc_device(size_t size, const device &dev,
                                  const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc_device(size_t size, const device &dev, const context &ctxt,
              const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *malloc_device(size_t size,
                                  const queue &q _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc_device(size_t size, const queue &q,
              const property_list &propList _CODELOCPARAM(&CodeLoc));

__SYCL_EXPORT void *
aligned_alloc_device(size_t alignment, size_t size, const device &dev,
                     const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_device(size_t alignment, size_t size, const device &dev,
                     const context &ctxt,
                     const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_device(size_t alignment, size_t size,
                     const queue &q _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_device(size_t alignment, size_t size, const queue &q,
                     const property_list &propList _CODELOCPARAM(&CodeLoc));

__SYCL_EXPORT void free(void *ptr, const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void free(void *ptr, const queue &q _CODELOCPARAM(&CodeLoc));

///
// Restricted USM
///
__SYCL_EXPORT void *malloc_host(size_t size,
                                const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc_host(size_t size, const context &ctxt,
            const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *malloc_host(size_t size,
                                const queue &q _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc_host(size_t size, const queue &q,
            const property_list &propList _CODELOCPARAM(&CodeLoc));

__SYCL_EXPORT void *malloc_shared(size_t size, const device &dev,
                                  const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc_shared(size_t size, const device &dev, const context &ctxt,
              const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *malloc_shared(size_t size,
                                  const queue &q _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc_shared(size_t size, const queue &q,
              const property_list &propList _CODELOCPARAM(&CodeLoc));

__SYCL_EXPORT void *
aligned_alloc_host(size_t alignment, size_t size,
                   const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_host(size_t alignment, size_t size, const context &ctxt,
                   const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *aligned_alloc_host(size_t alignment, size_t size,
                                       const queue &q _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_host(size_t alignment, size_t size, const queue &q,
                   const property_list &propList _CODELOCPARAM(&CodeLoc));

__SYCL_EXPORT void *
aligned_alloc_shared(size_t alignment, size_t size, const device &dev,
                     const context &ctxt _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_shared(size_t alignment, size_t size, const device &dev,
                     const context &ctxt,
                     const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_shared(size_t alignment, size_t size,
                     const queue &q _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc_shared(size_t alignment, size_t size, const queue &q,
                     const property_list &propList _CODELOCPARAM(&CodeLoc));

///
// single form
///
__SYCL_EXPORT void *malloc(size_t size, const device &dev, const context &ctxt,
                           usm::alloc kind _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc(size_t size, const device &dev, const context &ctxt, usm::alloc kind,
       const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *malloc(size_t size, const queue &q,
                           usm::alloc kind _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
malloc(size_t size, const queue &q, usm::alloc kind,
       const property_list &propList _CODELOCPARAM(&CodeLoc));

__SYCL_EXPORT void *aligned_alloc(size_t alignment, size_t size,
                                  const device &dev, const context &ctxt,
                                  usm::alloc kind _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc(size_t alignment, size_t size, const device &dev,
              const context &ctxt, usm::alloc kind,
              const property_list &propList _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *aligned_alloc(size_t alignment, size_t size, const queue &q,
                                  usm::alloc kind _CODELOCPARAM(&CodeLoc));
__SYCL_EXPORT void *
aligned_alloc(size_t alignment, size_t size, const queue &q, usm::alloc kind,
              const property_list &propList _CODELOCPARAM(&CodeLoc));

///
// Template forms
///
template <typename T>
T *malloc_device(
    size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc_device(alignof(T), Count * sizeof(T),
                                               Dev, Ctxt, PropList, CodeLoc));
}

template <typename T>
T *malloc_device(
    size_t Count, const queue &Q,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return malloc_device<T>(Count, Q.get_device(), Q.get_context(), PropList,
                          CodeLoc);
}

template <typename T>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc_device(max(Alignment, alignof(T)),
                                               Count * sizeof(T), Dev, Ctxt,
                                               PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc_device(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return aligned_alloc_device<T>(Alignment, Count, Q.get_device(),
                                 Q.get_context(), PropList, CodeLoc);
}

template <typename T>
T *malloc_host(
    size_t Count, const context &Ctxt,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc_host(alignof(T), Count * sizeof(T),
                                             Ctxt, PropList, CodeLoc));
}

template <typename T>
T *malloc_host(
    size_t Count, const queue &Q,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return malloc_host<T>(Count, Q.get_context(), PropList, CodeLoc);
}

template <typename T>
T *malloc_shared(
    size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc_shared(alignof(T), Count * sizeof(T),
                                               Dev, Ctxt, PropList, CodeLoc));
}

template <typename T>
T *malloc_shared(
    size_t Count, const queue &Q,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return malloc_shared<T>(Count, Q.get_device(), Q.get_context(), PropList,
                          CodeLoc);
}

template <typename T>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const context &Ctxt,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc_host(std ::max(Alignment, alignof(T)),
                                             Count * sizeof(T), Ctxt, PropList,
                                             CodeLoc));
}

template <typename T>
T *aligned_alloc_host(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return aligned_alloc_host<T>(Alignment, Count, Q.get_context(), PropList,
                               CodeLoc);
}

template <typename T>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc_shared(max(Alignment, alignof(T)),
                                               Count * sizeof(T), Dev, Ctxt,
                                               PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc_shared(
    size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return aligned_alloc_shared<T>(Alignment, Count, Q.get_device(),
                                 Q.get_context(), PropList, CodeLoc);
}

template <typename T>
T *malloc(
    size_t Count, const device &Dev, const context &Ctxt, usm::alloc Kind,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc(alignof(T), Count * sizeof(T), Dev,
                                        Ctxt, Kind, PropList, CodeLoc));
}

template <typename T>
T *malloc(
    size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return malloc<T>(Count, Q.get_device(), Q.get_context(), Kind, PropList,
                   CodeLoc);
}

template <typename T>
T *aligned_alloc(
    size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    usm::alloc Kind,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return static_cast<T *>(aligned_alloc(max(Alignment, alignof(T)),
                                        Count * sizeof(T), Dev, Ctxt, Kind,
                                        PropList, CodeLoc));
}

template <typename T>
T *aligned_alloc(
    size_t Alignment, size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {} _CODELOCPARAM(&CodeLoc)) {
  _CODELOCARG(&CodeLoc);
  return aligned_alloc<T>(Alignment, Count, Q.get_device(), Q.get_context(),
                          Kind, PropList, CodeLoc);
}

// Pointer queries
/// Query the allocation type from a USM pointer
///
/// \param ptr is the USM pointer to query
/// \param ctxt is the sycl context the ptr was allocated in
__SYCL_EXPORT usm::alloc get_pointer_type(const void *ptr, const context &ctxt);

/// Queries the device against which the pointer was allocated
/// Throws an invalid_object_error if ptr is a host allocation.
///
/// \param ptr is the USM pointer to query
/// \param ctxt is the sycl context the ptr was allocated in
__SYCL_EXPORT device get_pointer_device(const void *ptr, const context &ctxt);

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

//==---------------- usm.hpp - SYCL USM ------------------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/usm/usm_allocator.hpp>
#include <sycl/usm/usm_enums.hpp>

#include <cstddef>

#ifndef DISABLE_SYCL_INSTRUMENTATION_METADATA
#define __CODE_LOCATION_METADATA(...)                                          \
  (__VA_ARGS__,                                                                \
   const detail::code_location & = detail::code_location::current())
#define __START_CODE_LOCATION_METADATA_DEFAULT(CL, ...)                        \
  (__VA_ARGS__,                                                                \
   const detail::code_location &CL = detail::code_location::current()) {
#define __START_CODE_LOCATION_METADATA(CL, ...)                                \
  (__VA_ARGS__, const detail::code_location &CL) {
#else
#define __START_CODE_LOCATION_METADATA(CL, ...)                                \
  (__VA_ARGS__) {                                                              \
    const detail::code_location &CL = {};
#define __START_CODE_LOCATION_METADATA_DEFAULT(CL, ...)                        \
  __START_CODE_LOCATION_METADATA(CL, __VA_ARGS__)
#endif
#define __END_CODE_LOCATION_METADATA }

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
///
// Explicit USM
///
__SYCL_EXPORT void *malloc_device __CODE_LOCATION_METADATA(size_t size,
                                                           const device &dev,
                                                           const context &ctxt);
__SYCL_EXPORT void *malloc_device
__CODE_LOCATION_METADATA(size_t size, const device &dev, const context &ctxt,
                         const property_list &propList);
__SYCL_EXPORT void *malloc_device __CODE_LOCATION_METADATA(size_t size,
                                                           const queue &q);
__SYCL_EXPORT void *malloc_device __CODE_LOCATION_METADATA(
    size_t size, const queue &q, const property_list &propList);

__SYCL_EXPORT void *aligned_alloc_device __CODE_LOCATION_METADATA(
    size_t alignment, size_t size, const device &dev, const context &ctxt);
__SYCL_EXPORT void *aligned_alloc_device
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const device &dev,
                         const context &ctxt, const property_list &propList);
__SYCL_EXPORT void *aligned_alloc_device
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q);
__SYCL_EXPORT void *aligned_alloc_device
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q,
                         const property_list &propList);

__SYCL_EXPORT void free __CODE_LOCATION_METADATA(void *ptr,
                                                 const context &ctxt);
__SYCL_EXPORT void free __CODE_LOCATION_METADATA(void *ptr, const queue &q);

///
// Restricted USM
///
__SYCL_EXPORT void *malloc_host __CODE_LOCATION_METADATA(size_t size,
                                                         const context &ctxt);
__SYCL_EXPORT void *malloc_host __CODE_LOCATION_METADATA(
    size_t size, const context &ctxt, const property_list &propList);
__SYCL_EXPORT void *malloc_host __CODE_LOCATION_METADATA(size_t size,
                                                         const queue &q);
__SYCL_EXPORT void *malloc_host __CODE_LOCATION_METADATA(
    size_t size, const queue &q, const property_list &propList);

__SYCL_EXPORT void *malloc_shared __CODE_LOCATION_METADATA(size_t size,
                                                           const device &dev,
                                                           const context &ctxt);
__SYCL_EXPORT void *malloc_shared
__CODE_LOCATION_METADATA(size_t size, const device &dev, const context &ctxt,
                         const property_list &propList);
__SYCL_EXPORT void *malloc_shared __CODE_LOCATION_METADATA(size_t size,
                                                           const queue &q);
__SYCL_EXPORT void *malloc_shared __CODE_LOCATION_METADATA(
    size_t size, const queue &q, const property_list &propList);

__SYCL_EXPORT void *aligned_alloc_host
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const context &ctxt);
__SYCL_EXPORT void *aligned_alloc_host
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const context &ctxt,
                         const property_list &propList);
__SYCL_EXPORT void *aligned_alloc_host
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q);
__SYCL_EXPORT void *aligned_alloc_host
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q,
                         const property_list &propList);

__SYCL_EXPORT void *aligned_alloc_shared __CODE_LOCATION_METADATA(
    size_t alignment, size_t size, const device &dev, const context &ctxt);
__SYCL_EXPORT void *aligned_alloc_shared
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const device &dev,
                         const context &ctxt, const property_list &propList);
__SYCL_EXPORT void *aligned_alloc_shared
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q);
__SYCL_EXPORT void *aligned_alloc_shared
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q,
                         const property_list &propList);

///
// single form
///
__SYCL_EXPORT void *malloc __CODE_LOCATION_METADATA(size_t size,
                                                    const device &dev,
                                                    const context &ctxt,
                                                    usm::alloc kind);
__SYCL_EXPORT void *malloc
__CODE_LOCATION_METADATA(size_t size, const device &dev, const context &ctxt,
                         usm::alloc kind, const property_list &propList);
__SYCL_EXPORT void *malloc __CODE_LOCATION_METADATA(size_t size, const queue &q,
                                                    usm::alloc kind);
__SYCL_EXPORT void *malloc
__CODE_LOCATION_METADATA(size_t size, const queue &q, usm::alloc kind,
                         const property_list &propList);

__SYCL_EXPORT void *aligned_alloc __CODE_LOCATION_METADATA(size_t alignment,
                                                           size_t size,
                                                           const device &dev,
                                                           const context &ctxt,
                                                           usm::alloc kind);
__SYCL_EXPORT void *aligned_alloc __CODE_LOCATION_METADATA(
    size_t alignment, size_t size, const device &dev, const context &ctxt,
    usm::alloc kind, const property_list &propList);
__SYCL_EXPORT void *aligned_alloc __CODE_LOCATION_METADATA(size_t alignment,
                                                           size_t size,
                                                           const queue &q,
                                                           usm::alloc kind);
__SYCL_EXPORT void *aligned_alloc
__CODE_LOCATION_METADATA(size_t alignment, size_t size, const queue &q,
                         usm::alloc kind, const property_list &propList);

///
// Template forms
///
template <typename T>
T *malloc_device __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc_device(alignof(T), Count * sizeof(T),
                                               Dev, Ctxt, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc_device __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const queue &Q, const property_list &PropList = {}) {
  return malloc_device<T>(Count, Q.get_device(), Q.get_context(), PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc_device __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc_device(
      max(Alignment, alignof(T)), Count * sizeof(T), Dev, Ctxt, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc_device __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {}) {
  return aligned_alloc_device<T>(Alignment, Count, Q.get_device(),
                                 Q.get_context(), PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc_host __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const context &Ctxt, const property_list &PropList = {}) {
  return static_cast<T *>(
      aligned_alloc_host(alignof(T), Count * sizeof(T), Ctxt, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc_host __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const queue &Q, const property_list &PropList = {}) {
  return malloc_host<T>(Count, Q.get_context(), PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc_shared __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc_shared(alignof(T), Count * sizeof(T),
                                               Dev, Ctxt, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc_shared __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const queue &Q, const property_list &PropList = {}) {
  return malloc_shared<T>(Count, Q.get_device(), Q.get_context(), PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc_host __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const context &Ctxt,
    const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc_host(
      std ::max(Alignment, alignof(T)), Count * sizeof(T), Ctxt, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc_host __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {}) {
  return aligned_alloc_host<T>(Alignment, Count, Q.get_context(), PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc_shared __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc_shared(
      max(Alignment, alignof(T)), Count * sizeof(T), Dev, Ctxt, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc_shared __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const queue &Q,
    const property_list &PropList = {}) {
  return aligned_alloc_shared<T>(Alignment, Count, Q.get_device(),
                                 Q.get_context(), PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const device &Dev, const context &Ctxt, usm::alloc Kind,
    const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc(alignof(T), Count * sizeof(T), Dev,
                                        Ctxt, Kind, PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *malloc __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {}) {
  return malloc<T>(Count, Q.get_device(), Q.get_context(), Kind, PropList, CL);
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const device &Dev, const context &Ctxt,
    usm::alloc Kind, const property_list &PropList = {}) {
  return static_cast<T *>(aligned_alloc(max(Alignment, alignof(T)),
                                        Count * sizeof(T), Dev, Ctxt, Kind,
                                        PropList, CL));
}
__END_CODE_LOCATION_METADATA

template <typename T>
T *aligned_alloc __START_CODE_LOCATION_METADATA_DEFAULT(
    CL, size_t Alignment, size_t Count, const queue &Q, usm::alloc Kind,
    const property_list &PropList = {}) {
  return aligned_alloc<T>(Alignment, Count, Q.get_device(), Q.get_context(),
                          Kind, PropList, CL);
}
__END_CODE_LOCATION_METADATA

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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

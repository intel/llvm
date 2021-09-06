//==---------------- reduction.hpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/known_identity.hpp>

#include "sycl/ext/oneapi/reduction.hpp"

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction operation \p Combiner, and optional reduction properties.
template <typename T, typename AllocatorT, typename BinaryOperation>
std::enable_if_t<has_known_identity<BinaryOperation, T>::value,
                 ext::oneapi::detail::reduction_impl<
                     T, BinaryOperation, 1, false, access::placeholder::true_t>>
reduction(buffer<T, 1, AllocatorT> Var, handler &CGH, BinaryOperation,
          const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return {Var, CGH, InitializeToIdentity};
}

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction operation \p Combiner, and optional reduction properties.
/// The reduction algorithm may be less efficient for this variant as the
/// reduction identity is not known statically and it is not provided by user.
template <typename T, typename AllocatorT, typename BinaryOperation>
std::enable_if_t<!has_known_identity<BinaryOperation, T>::value,
                 ext::oneapi::detail::reduction_impl<
                     T, BinaryOperation, 1, false, access::placeholder::true_t>>
reduction(buffer<T, 1, AllocatorT>, handler &, BinaryOperation,
          const property_list &PropList = {}) {
  // TODO: implement reduction that works even when identity is not known.
  (void)PropList;
  throw runtime_error("Identity-less reductions with unknown identity are not "
                      "supported yet.",
                      PI_INVALID_VALUE);
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given USM pointer \p Var, handler \p CGH, reduction operation
/// \p Combiner, and optional reduction properties.
template <typename T, typename BinaryOperation>
std::enable_if_t<
    has_known_identity<BinaryOperation, T>::value,
    ext::oneapi::detail::reduction_impl<T, BinaryOperation, 1, true>>
reduction(T *Var, BinaryOperation, const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return {Var, InitializeToIdentity};
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given USM pointer \p Var, handler \p CGH, reduction operation
/// \p Combiner, and optional reduction properties.
/// The reduction algorithm may be less efficient for this variant as the
/// reduction identity is not known statically and it is not provided by user.
template <typename T, typename BinaryOperation>
std::enable_if_t<
    !has_known_identity<BinaryOperation, T>::value,
    ext::oneapi::detail::reduction_impl<T, BinaryOperation, 1, true>>
reduction(T *, BinaryOperation, const property_list &PropList = {}) {
  // TODO: implement reduction that works even when identity is not known.
  (void)PropList;
  throw runtime_error("Identity-less reductions with unknown identity are not "
                      "supported yet.",
                      PI_INVALID_VALUE);
}

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction identity value \p Identity, reduction operation \p Combiner,
/// and optional reduction properties.
template <typename T, typename AllocatorT, typename BinaryOperation>
ext::oneapi::detail::reduction_impl<T, BinaryOperation, 1, false,
                                    access::placeholder::true_t>
reduction(buffer<T, 1, AllocatorT> Var, handler &CGH, const T &Identity,
          BinaryOperation Combiner, const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return {Var, CGH, Identity, Combiner, InitializeToIdentity};
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given USM pointer \p Var, reduction identity value \p Identity,
/// binary operation \p Combiner, and optional reduction properties.
template <typename T, typename BinaryOperation>
ext::oneapi::detail::reduction_impl<T, BinaryOperation, 1, true>
reduction(T *Var, const T &Identity, BinaryOperation Combiner,
          const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return {Var, Identity, Combiner, InitializeToIdentity};
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

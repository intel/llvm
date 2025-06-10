//===- impl_utils.hpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>     // for assert
#include <type_traits> // for add_pointer_t
#include <utility>     // for forward

namespace sycl {
inline namespace _V1 {
namespace detail {

// Helper function for extracting implementation from SYCL's interface objects.
// Note! This function relies on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from this function.
//
// Note that due to a bug in MSVC compilers (including MSVC2019 v19.20), it
// may not recognize the usage of this function in friend member declarations
// if the template parameter name there is not equal to the name used here,
// i.e. 'Obj'. For example, using 'Obj' here and 'T' in such declaration
// would trigger that error in MSVC:
//   template <class T>
//   friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);
template <class Obj>
const decltype(Obj::impl) &getSyclObjImpl(const Obj &SyclObject) {
  assert(SyclObject.impl && "every constructor should create an impl");
  return SyclObject.impl;
}

// Helper function for creation SYCL interface objects from implementations.
// Note! These functions rely on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from these functions.
template <class T>
T createSyclObjFromImpl(
    std::add_rvalue_reference_t<decltype(T::impl)> ImplObj) {
  return T(std::forward<decltype(ImplObj)>(ImplObj));
}

template <class T>
T createSyclObjFromImpl(
    std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj) {
  return T(ImplObj);
}

template <class T>
T createSyclObjFromImpl(
    std::add_lvalue_reference_t<typename std::remove_reference_t<
        decltype(getSyclObjImpl(std::declval<T>()))>::element_type>
        ImplRef) {
  return createSyclObjFromImpl<T>(ImplRef.shared_from_this());
}

} // namespace detail
} // namespace _V1
} // namespace sycl

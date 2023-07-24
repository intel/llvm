//===- impl_utils.hpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <cassert>

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
template <class Obj> decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject) {
  assert(SyclObject.impl && "every constructor should create an impl");
  return SyclObject.impl;
}

// Returns the raw pointer to the impl object of given face object. The caller
// must make sure the returned pointer is not captured in a field or otherwise
// stored - i.e. must live only as on-stack value.
template <class T>
typename std::add_pointer_t<typename decltype(T::impl)::element_type>
getRawSyclObjImpl(const T &SyclObject) {
  return SyclObject.impl.get();
}

// Helper function for creation SYCL interface objects from implementations.
// Note! This function relies on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from this function.
template <class T> T createSyclObjFromImpl(decltype(T::impl) ImplObj) {
  return T(ImplObj);
}

} // namespace detail
} // namespace _V1
} // namespace sycl

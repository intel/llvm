//==----------- spec_constant.hpp - SYCL public oneapi API header file -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Based on:
// https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md
// TODO:
// 1) implement the SPIR-V interop part of the proposal
// 2) move to the new upcoming spec constant specification (then 1 above is not
//    needed)

#pragma once

#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/detail/sycl_fe_intrins.hpp>
#include <sycl/exception.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext::oneapi::experimental {

class spec_const_error : public compile_program_error {
  using compile_program_error::compile_program_error;
};

template <typename T, typename ID = T>
class __SYCL2020_DEPRECATED(
    "Specialization constats extension is deprecated, use SYCL 2020"
    " specialization constants instead")
    __SYCL_TYPE(spec_constant) spec_constant {
public:
  spec_constant() {}

private:
#ifndef __SYCL_DEVICE_ONLY__
  // Implementation defined constructor.
  spec_constant(T Cst) : Val(Cst) {}

  T Val;
#else
  char padding[sizeof(T)];
#endif // __SYCL_DEVICE_ONLY__

public:
  template <typename V = T>
  typename sycl::detail::enable_if_t<std::is_arithmetic<V>::value, V>
  get() const { // explicit access.
#ifdef __SYCL_DEVICE_ONLY__
    const char *TName = __builtin_sycl_unique_stable_name(ID);
    return __sycl_getScalarSpecConstantValue<T>(TName);
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename V = T>
  typename sycl::detail::enable_if_t<std::is_class<V>::value &&
                                         std::is_pod<V>::value,
                                     V>
  get() const { // explicit access.
#ifdef __SYCL_DEVICE_ONLY__
    const char *TName = __builtin_sycl_unique_stable_name(ID);
    return __sycl_getCompositeSpecConstantValue<T>(TName);
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }

  operator T() const { // implicit conversion.
    return get();
  }
};

} // namespace ext::oneapi::experimental

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

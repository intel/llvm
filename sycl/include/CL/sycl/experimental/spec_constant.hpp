//==----- spec_constant.hpp - SYCL public experimental API header file -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Based on:
// https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md
// TODO:
// 1) implement the SPIRV interop part of the proposal
// 2) support arbitrary spec constant type; only primitive types are
//    supported currently
// 3) move to the new upcoming spec constant specification (then 1 above is not
//    needed)

#pragma once

#include <CL/sycl/detail/sycl_fe_intrins.hpp>
#include <CL/sycl/exception.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace experimental {

class spec_const_error : public compile_program_error {};

template <typename T, typename ID = T> class spec_constant {
private:
  // Implementation defined constructor.
#ifdef __SYCL_DEVICE_ONLY__
  spec_constant() {}
#else
  spec_constant(T Cst) : Val(Cst) {}
#endif
#ifndef __SYCL_DEVICE_ONLY__
  T Val;
#endif
  friend class cl::sycl::program;

public:
  T get() const { // explicit access.
#ifdef __SYCL_DEVICE_ONLY__
    const char *TName = __builtin_unique_stable_name(ID);
    return __sycl_getSpecConstantValue<T>(TName);
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }

  operator T() const { // implicit conversion.
    return get();
  }
};

} // namespace experimental
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

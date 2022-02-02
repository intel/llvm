//==------ kernel_handler.hpp -- SYCL standard header file -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/sycl_fe_intrins.hpp>
#include <CL/sycl/exception.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
/// Reading the value of a specialization constant
///
/// \ingroup sycl_api
class kernel_handler {
public:
#if __cplusplus >= 201703L
  template <auto &S>
  typename std::remove_reference_t<decltype(S)>::value_type
  get_specialization_constant() {
#ifdef __SYCL_DEVICE_ONLY__
    return getSpecializationConstantOnDevice<S>();
#else
    // TODO: add support of host device
    throw cl::sycl::feature_not_supported(
        "kernel_handler::get_specialization_constant() is not yet supported by "
        "host device.",
        PI_INVALID_OPERATION);
#endif // __SYCL_DEVICE_ONLY__
  }
#endif // __cplusplus >= 201703L

private:
  void __init_specialization_constants_buffer(
      char *SpecializationConstantsBuffer = nullptr) {
    MSpecializationConstantsBuffer = SpecializationConstantsBuffer;
  }

#ifdef __SYCL_DEVICE_ONLY__
#if __cplusplus >= 201703L
  template <
      auto &S,
      typename T = typename std::remove_reference_t<decltype(S)>::value_type,
      std::enable_if_t<std::is_fundamental_v<T>> * = nullptr>
  T getSpecializationConstantOnDevice() {
    const char *SymbolicID = __builtin_sycl_unique_stable_id(S);
    return __sycl_getScalar2020SpecConstantValue<T>(
        SymbolicID, &S, MSpecializationConstantsBuffer);
  }
  template <
      auto &S,
      typename T = typename std::remove_reference_t<decltype(S)>::value_type,
      std::enable_if_t<std::is_compound_v<T>> * = nullptr>
  T getSpecializationConstantOnDevice() {
    const char *SymbolicID = __builtin_sycl_unique_stable_id(S);
    return __sycl_getComposite2020SpecConstantValue<T>(
        SymbolicID, &S, MSpecializationConstantsBuffer);
  }
#endif // __cplusplus >= 201703L
#endif // __SYCL_DEVICE_ONLY__

  char *MSpecializationConstantsBuffer = nullptr;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

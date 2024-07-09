//==------ kernel_handler.hpp -- SYCL standard header file -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>            // for __SYCL_TYPE
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/pi.h>                   // for PI_ERROR_INVALID_OPERATION
#include <sycl/exception.hpp>                 // for feature_not_supported

#ifdef __SYCL_DEVICE_ONLY__
#include <CL/__spirv/spirv_ops.hpp>
#endif

#include <type_traits> // for remove_reference_t

#ifdef __SYCL_DEVICE_ONLY__
// Get the value of the specialization constant with given symbolic ID.
// `SymbolicID` is a unique string ID of a specialization constant.
// `DefaultValue` contains a pointer to a global variable with the initializer,
// which should be used as the default value of the specialization constants.
// `RTBuffer` is a pointer to a runtime buffer, which holds values of all
// specialization constant and should be used if native specialization constants
// are not available.
template <typename T>
__DPCPP_SYCL_EXTERNAL T __sycl_getScalar2020SpecConstantValue(
    const char *SymbolicID, const void *DefaultValue, const void *RTBuffer);

template <typename T>
__DPCPP_SYCL_EXTERNAL T __sycl_getComposite2020SpecConstantValue(
    const char *SymbolicID, const void *DefaultValue, const void *RTBuffer);
#endif

namespace sycl {
inline namespace _V1 {
/// Reading the value of a specialization constant
///
/// \ingroup sycl_api
class __SYCL_TYPE(kernel_handler) kernel_handler {
public:
  template <auto &S>
  __SYCL_ALWAYS_INLINE typename std::remove_reference_t<decltype(S)>::value_type
  get_specialization_constant() {
#ifdef __SYCL_DEVICE_ONLY__
    return getSpecializationConstantOnDevice<S>();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "kernel_handler::get_specialization_constant() is "
                          "not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

private:
  void __init_specialization_constants_buffer(
      char *SpecializationConstantsBuffer = nullptr) {
    MSpecializationConstantsBuffer = SpecializationConstantsBuffer;
  }

#ifdef __SYCL_DEVICE_ONLY__
  template <
      auto &S,
      typename T = typename std::remove_reference_t<decltype(S)>::value_type,
      std::enable_if_t<std::is_scalar_v<T>> * = nullptr>
  __SYCL_ALWAYS_INLINE T getSpecializationConstantOnDevice() {
    const char *SymbolicID = __builtin_sycl_unique_stable_id(S);
    return __sycl_getScalar2020SpecConstantValue<T>(
        SymbolicID, &S, MSpecializationConstantsBuffer);
  }
  template <
      auto &S,
      typename T = typename std::remove_reference_t<decltype(S)>::value_type,
      std::enable_if_t<!std::is_scalar_v<T>> * = nullptr>
  __SYCL_ALWAYS_INLINE T getSpecializationConstantOnDevice() {
    const char *SymbolicID = __builtin_sycl_unique_stable_id(S);
    return __sycl_getComposite2020SpecConstantValue<T>(
        SymbolicID, &S, MSpecializationConstantsBuffer);
  }
#endif // __SYCL_DEVICE_ONLY__

  char *MSpecializationConstantsBuffer = nullptr;
};

} // namespace _V1
} // namespace sycl

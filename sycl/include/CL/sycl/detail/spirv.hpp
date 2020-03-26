//===-- spirv.hpp - Helpers to generate SPIR-V instructions ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/type_traits.hpp>

#ifdef __SYCL_DEVICE_ONLY__
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace spirv {

// Broadcast with scalar local index
template <__spv::Scope::Flag S, typename T, typename IdT>
detail::enable_if_t<std::is_integral<IdT>::value, T>
GroupBroadcast(T x, IdT local_id) {
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<IdT>;
  OCLT ocl_x = detail::convertDataToType<T, OCLT>(x);
  OCLIdT ocl_id = detail::convertDataToType<IdT, OCLIdT>(local_id);
  return __spirv_GroupBroadcast(S, ocl_x, ocl_id);
}

// Broadcast with vector local index
template <__spv::Scope::Flag S, typename T, int Dimensions>
T GroupBroadcast(T x, id<Dimensions> local_id) {
  if (Dimensions == 1) {
    return GroupBroadcast<S>(x, local_id[0]);
  }
  using IdT = vec<size_t, Dimensions>;
  using OCLT = detail::ConvertToOpenCLType_t<T>;
  using OCLIdT = detail::ConvertToOpenCLType_t<IdT>;
  IdT vec_id;
  for (int i = 0; i < Dimensions; ++i) {
    vec_id[i] = local_id[Dimensions - i - 1];
  }
  OCLT ocl_x = detail::convertDataToType<T, OCLT>(x);
  OCLIdT ocl_id = detail::convertDataToType<IdT, OCLIdT>(vec_id);
  return __spirv_GroupBroadcast(S, ocl_x, ocl_id);
}

} // namespace spirv
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif //  __SYCL_DEVICE_ONLY__

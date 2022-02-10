//==--------- cuda.hpp - SYCL CUDA backend ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend.hpp>
#include <CL/sycl/context.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// CUDA context specialization
template <>
auto get_native<backend::ext_oneapi_cuda, context>(const context &C)
    -> backend_return_t<backend::ext_oneapi_cuda, context> {
  // create a vector to be returned
  backend_return_t<backend::ext_oneapi_cuda, context> ret;

  // get the native CUDA context from the SYCL object
  auto native = reinterpret_cast<
      backend_return_t<backend::ext_oneapi_cuda, context>::value_type>(
      C.getNative());
  ret.push_back(native);

  return ret;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

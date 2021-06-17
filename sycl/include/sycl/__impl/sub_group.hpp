//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/ONEAPI/sub_group.hpp>
#include <sycl/__impl/group.hpp>

namespace __sycl_internal {
inline namespace __v1 {
using ONEAPI::sub_group;
// TODO move the entire sub_group class implementation to this file once
// breaking changes are allowed.
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
#endif
namespace sycl {
  using namespace __sycl_internal::__v1;
}
#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
}
#endif

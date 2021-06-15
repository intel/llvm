//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/ONEAPI/sub_group.hpp>
#include <CL/sycl/group.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
using ONEAPI::sub_group;
// TODO move the entire sub_group class implementation to this file once
// breaking changes are allowed.
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

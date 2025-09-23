//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/kernel_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// The structure represents kernel argument.
class ArgDesc {
public:
  ArgDesc(sycl::detail::kernel_param_kind_t Type, void *Ptr, int Size,
          int Index)
      : MType(Type), MPtr(Ptr), MSize(Size), MIndex(Index) {}

  sycl::detail::kernel_param_kind_t MType;
  void *MPtr;
  int MSize;
  int MIndex;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
//==--------------------- abi_check.hpp - SYCL ABI checks ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#define __SYCL_ABI_PACKING_ERROR                                               \
  "Structure packing is incompatible with the supported SYCL runtime ABI. "    \
  "Remove -fpack-struct or -Zp, and do not include SYCL headers inside "       \
  "#pragma pack. Apply packing only to user-defined types instead."

#ifndef __SYCL_DEVICE_ONLY__
namespace sycl {
inline namespace _V1 {
namespace detail {
// This header is included by kernel_desc.hpp, which cannot include standard
// library headers. Use only built-in types to detect incompatible host packing.
struct abi_packing_check {
  char Byte;
  void *Pointer;
};
static_assert(alignof(abi_packing_check) == alignof(void *),
              __SYCL_ABI_PACKING_ERROR);
} // namespace detail
} // namespace _V1
} // namespace sycl
#endif

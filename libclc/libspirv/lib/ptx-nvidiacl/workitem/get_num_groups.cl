//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <libspirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

_CLC_DEF _CLC_OVERLOAD size_t __spirv_NumWorkgroups_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}

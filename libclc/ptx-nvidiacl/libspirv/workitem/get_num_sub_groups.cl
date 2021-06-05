//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

_CLC_DEF _CLC_OVERLOAD uint __spirv_NumSubgroups() {
  // sreg.nwarpid returns number of warp identifiers, not number of warps
  // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  size_t size_x = __spirv_WorkgroupSize_x();
  size_t size_y = __spirv_WorkgroupSize_y();
  size_t size_z = __spirv_WorkgroupSize_z();
  uint sg_size = __spirv_SubgroupMaxSize();
  uint linear_size = size_z * size_y * size_x;
  return (linear_size + sg_size - 1) / sg_size;
}

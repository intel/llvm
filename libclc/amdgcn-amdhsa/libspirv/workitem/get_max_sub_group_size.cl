//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// FIXME: Remove the following workaround once the clang change is released.
// This is for backward compatibility with older clang which does not define
// __AMDGCN_WAVEFRONT_SIZE. It does not consider -mwavefrontsize64.
// See:
// https://github.com/intel/llvm/blob/sycl/clang/lib/Basic/Targets/AMDGPU.h#L414
// and:
// https://github.com/intel/llvm/blob/sycl/clang/lib/Basic/Targets/AMDGPU.cpp#L421
#ifndef __AMDGCN_WAVEFRONT_SIZE
#if __gfx1010__ || __gfx1011__ || __gfx1012__ || __gfx1030__ || __gfx1031__
#define __AMDGCN_WAVEFRONT_SIZE 32
#else
#define __AMDGCN_WAVEFRONT_SIZE 64
#endif
#endif

_CLC_DEF _CLC_OVERLOAD uint __spirv_SubgroupMaxSize() {
  return __AMDGCN_WAVEFRONT_SIZE;
}

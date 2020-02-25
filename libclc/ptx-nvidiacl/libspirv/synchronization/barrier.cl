//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.
_CLC_DEF void _Z22__spirv_ControlBarrierN5__spv5ScopeES0_j(enum Scope scope, enum Scope memory, unsigned int semantics) {
  __syncthreads();
}

// TODO: Stop manually mangling this name. Need C++ namespaces to get the exact mangling.
_CLC_DEF void _Z21__spirv_MemoryBarrierN5__spv5ScopeEj(enum Scope scope, unsigned int semantics) {
}

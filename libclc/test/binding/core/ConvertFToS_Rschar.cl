//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clang %link-libspirv -emit-llvm -S -o - %s | FileCheck %s
// RUN: %clang %link-remangled-libspirv -DSKIP_FP16 -emit-llvm -S -o - %s | FileCheck %s

#include <libspirv/spirv_types.h>

// CHECK-NOT: declare {{.*}} @_Z
// CHECK-NOT: call {{[^ ]*}} bitcast
__attribute__((overloadable)) __clc_schar_t
test___spirv_ConvertFToS_Rschar(__clc_fp32_t args_0) {
  return __spirv_ConvertFToS_Rschar(args_0);
}

#ifdef cl_khr_fp64
__attribute__((overloadable)) __clc_schar_t
test___spirv_ConvertFToS_Rschar(__clc_fp64_t args_0) {
  return __spirv_ConvertFToS_Rschar(args_0);
}

#endif
#if defined(cl_khr_fp16) && !defined(SKIP_FP16)
__attribute__((overloadable)) __clc_schar_t
test___spirv_ConvertFToS_Rschar(__clc_fp16_t args_0) {
  return __spirv_ConvertFToS_Rschar(args_0);
}

#endif

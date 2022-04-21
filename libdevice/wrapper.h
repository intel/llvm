//==--- wrapper.h - declarations for devicelib functions -----*- C++ -*-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBDEVICE_WRAPPER_H__
#define __LIBDEVICE_WRAPPER_H__

#include "device.h"

#ifdef __SPIR__

#include <cstddef>
#include <cstdint>
// We need the following header to ensure the definition of all spirv variables
// required by the wrapper libraries.
#include "spirv_vars.h"
DEVICE_EXTERN_C
void *__devicelib_memcpy(void *dest, const void *src, size_t n);
DEVICE_EXTERN_C
void *__devicelib_memset(void *dest, int c, size_t n);
DEVICE_EXTERN_C
int __devicelib_memcmp(const void *s1, const void *s2, size_t n);
DEVICE_EXTERN_C
void __devicelib_assert_fail(const char *expr, const char *file, int32_t line,
                             const char *func, uint64_t gid0, uint64_t gid1,
                             uint64_t gid2, uint64_t lid0, uint64_t lid1,
                             uint64_t lid2);
#endif // __SPIR__
#endif // __LIBDEVICE_WRAPPER_H__

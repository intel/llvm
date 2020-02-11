//==--- wrapper.h - declarations for devicelib functions -----*- C++ -*-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_WRAPPERS_H__
#define __SYCL_WRAPPERS_H__

#include <cstddef>
#include <cstdint>

SYCL_EXTERNAL
extern "C" void __devicelib_assert_fail(
    const char *expr, const char *file,
    int32_t line, const char *func,
    uint64_t gid0, uint64_t gid1, uint64_t gid2,
    uint64_t lid0, uint64_t lid1, uint64_t lid2);

#endif // __SYCL_WRAPPERS_H__

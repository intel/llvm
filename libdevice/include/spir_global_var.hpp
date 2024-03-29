//==- spir_global_var.hpp - Declaration for device global variable support -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header

#define __SYCL_GLOBAL__ __attribute__((opencl_global))
#define __SYCL_LOCAL__ __attribute__((opencl_local))
#define __SYCL_PRIVATE__ __attribute__((opencl_private))
#define __SYCL_CONSTANT__ __attribute__((opencl_constant))

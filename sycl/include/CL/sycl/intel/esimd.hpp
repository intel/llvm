//==------------ esimd.hpp - DPC++ Explicit SIMD API -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of the Explicit SIMD API.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/intel/esimd/esimd.hpp>
#include <CL/sycl/intel/esimd/esimd_math.hpp>
#include <CL/sycl/intel/esimd/esimd_memory.hpp>
#include <CL/sycl/intel/esimd/esimd_view.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_ESIMD_KERNEL __attribute__((sycl_explicit_simd))
#define SYCL_ESIMD_FUNCTION __attribute__((sycl_explicit_simd))
#else
#define SYCL_ESIMD_KERNEL
#define SYCL_ESIMD_FUNCTION
#endif

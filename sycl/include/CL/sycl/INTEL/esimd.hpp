//==------------ esimd.hpp - DPC++ Explicit SIMD API redirection header ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header is deprecated and should not be included by applications.
// The header redirected to below should be used instead.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

__SYCL_WARNING("CL/sycl/INTEL/esimd.hpp usage is deprecated, include "
               "sycl/ext/intel/experimental/esimd.hpp instead")

#include <sycl/ext/intel/experimental/esimd.hpp>

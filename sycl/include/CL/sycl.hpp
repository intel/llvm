//==------------ sycl.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __SYCL2020_HEADER_INCLUDED
#error "Including both <CL/sycl.hpp> and <sycl/sycl.hpp> is not allowed"
#endif

#define __SYCL_ENABLE_SYCL121_NAMESPACE

#include <sycl/sycl.hpp>


//==------ sycl_fe_intrins.hpp --- SYCL Device Compiler's FE intrinsics ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// C++ intrinsics recognized by the SYCL device compiler frontend
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

// Get the value of the specialization constant with given name.
// Post-link tool traces the ID to a string literal it points to and assigns
// integer ID.
template <typename T>
SYCL_EXTERNAL T __sycl_getSpecConstantValue(const char *ID);

#endif

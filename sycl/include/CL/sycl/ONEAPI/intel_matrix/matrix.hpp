//==------------------ matrix.hpp - SYCL matrix ----------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
/// -DAMX will enable joint_matrix feature for AMX
///
// ===--------------------------------------------------------------------=== //

#pragma once

#if defined(__AMXTILE__) && defined(__AMXINT8__) && defined(__AMXBF16__)
#include <CL/sycl/ONEAPI/intel_matrix/matrix-amx.hpp>
#endif

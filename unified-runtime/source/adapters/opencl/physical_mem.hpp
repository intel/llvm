//===---------- physical_mem.hpp - OpenCL Adapter -------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"

namespace ur::opencl {

/// UR queue mapping on physical memory allocations used in virtual memory
/// management.
/// TODO: Implement.
///
struct ur_physical_mem_handle_t_ : handle_base {};

} // namespace ur::opencl

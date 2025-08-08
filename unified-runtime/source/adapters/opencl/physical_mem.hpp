//===---------- physical_mem.hpp - OpenCL Adapter -------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"

/// UR queue mapping on physical memory allocations used in virtual memory
/// management.
/// TODO: Implement.
///
struct ur_physical_mem_handle_t_ : ur::opencl::handle_base {};

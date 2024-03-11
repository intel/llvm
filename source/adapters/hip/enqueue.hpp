//===--------- event.hpp - HIP Adapter ------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

void simpleGuessLocalWorkSize(size_t *ThreadsPerBlock,
                              const size_t *GlobalWorkSize,
                              const size_t MaxThreadsPerBlock[3],
                              ur_kernel_handle_t Kernel);

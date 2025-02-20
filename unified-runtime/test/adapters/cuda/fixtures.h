// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_TEST_CONFORMANCE_ADAPTERS_CUDA_FIXTURES_H_INCLUDED
#define UR_TEST_CONFORMANCE_ADAPTERS_CUDA_FIXTURES_H_INCLUDED
#include <cuda.h>
#include <uur/fixtures.h>

#ifndef ASSERT_SUCCESS_CUDA
#define ASSERT_SUCCESS_CUDA(ACTUAL) ASSERT_EQ(CUDA_SUCCESS, (ACTUAL))
#endif // ASSERT_SUCCESS_CUDA

#ifndef EXPECT_SUCCESS_CUDA
#define EXPECT_SUCCESS_CUDA(ACTUAL) EXPECT_EQ(CUDA_SUCCESS, (ACTUAL))
#endif // EXPECT_EQ_RESULT_CUDA

#endif // UR_TEST_CONFORMANCE_ADAPTERS_CUDA_FIXTURES_H_INCLUDED

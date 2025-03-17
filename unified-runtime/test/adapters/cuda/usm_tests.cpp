// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

// We set the devicemap flag by default for...reasons. See implementation
// comment for details
static constexpr unsigned DEFAULT_EXPECTED = CU_MEMHOSTALLOC_DEVICEMAP;

uur::USMFlagsTestParams HostFlagPairs[] = {
    // Of the USM flags CUDA is only capable of supporting the WRITE_COMBINE
    // flag so we check for that here
    {UR_USM_HOST_MEM_FLAG_WRITE_COMBINE,
     CU_MEMHOSTALLOC_WRITECOMBINED | DEFAULT_EXPECTED},
};

struct CUDAUSMFlagsTest : uur::USMFlagsTest {
  virtual unsigned getAllocationFlags(void *mem) override {
    unsigned attrs{};
    EXPECT_EQ(cuMemHostGetFlags(&attrs, mem), CUDA_SUCCESS);
    return attrs;
  }
};

TEST_P(CUDAUSMFlagsTest, HostFlagPairs) { testParamWithHostAlloc(); };

UUR_PLATFORM_TEST_SUITE_WITH_PARAM(
    CUDAUSMFlagsTest, testing::ValuesIn(HostFlagPairs),
    uur::platformTestWithParamPrinter<uur::USMFlagsTestParams>);

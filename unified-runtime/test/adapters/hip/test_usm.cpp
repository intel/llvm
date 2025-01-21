// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <hip/hip_runtime_api.h>

uur::USMFlagsTestParams HostFlagPairs[] = {
    {UR_USM_HOST_MEM_FLAG_HOST_COHERENT, hipHostMallocCoherent},
    {UR_USM_HOST_MEM_FLAG_HOST_NON_COHERENT, hipHostMallocNonCoherent},
    // Of the USM flags HIP supports coherent and non-coherent flags.
    // The API docs advertise support for cuda's
    // UR_USM_HOST_MEM_FLAG_WRITE_COMBINE namesake,
    // CU_MEMHOSTALLOC_WRITECOMBINED, but it is ignored and we don't try and set
    // it. See https://github.com/ROCm/clr/issues/114 for context.
    // Update this here if anything changes
    {UR_USM_HOST_MEM_FLAG_WRITE_COMBINE, 0},
};

struct HipUSMFlagsTest : uur::USMFlagsTest {
  virtual unsigned getAllocationFlags(void *mem) override {
    hipPointerAttribute_t attrs{};
    EXPECT_EQ(hipPointerGetAttributes(&attrs, mem), HIP_SUCCESS);
    return (unsigned long)attrs.allocationFlags;
  }
};

TEST_P(HipUSMFlagsTest, HostFlagPairs) { testParamWithHostAlloc(); }

UUR_PLATFORM_TEST_SUITE_WITH_PARAM(
    HipUSMFlagsTest, testing::ValuesIn(HostFlagPairs),
    uur::platformTestWithParamPrinter<uur::USMFlagsTestParams>);

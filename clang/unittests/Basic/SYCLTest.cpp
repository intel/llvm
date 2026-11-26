//===- unittests/Basic/SYCLTest.cpp - Test SYCL offloading utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SYCL.h"
#include "gtest/gtest.h"

using namespace clang;

static OffloadArch parse(llvm::StringRef S) { return StringToOffloadArch(S); }

TEST(SYCLTest, SupportedAMDGPUArch) {
  EXPECT_TRUE(IsSYCLSupportedAMDGPUArch(parse("gfx906")));
  EXPECT_TRUE(IsSYCLSupportedAMDGPUArch(parse("gfx1201")));
  // Generic AMD GPUs and AMDGPU pseudo targets are not supported.
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("gfx9-generic")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("gfx10-3-generic")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("gfx12-generic")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("generic-hsa")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("amdgcnspirv")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("sm_80")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("pvc")));
  EXPECT_FALSE(IsSYCLSupportedAMDGPUArch(parse("unknown_arch")));
}

TEST(SYCLTest, SupportedNVidiaGPUArch) {
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(parse("sm_50")));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(parse("sm_80")));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(parse("sm_90")));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(parse("sm_90a")));
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(parse("sm_35")));
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(parse("sm_100")));
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(parse("gfx906")));
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(parse("bmg_g21")));
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(parse("unknown_arch")));
}

//===- unittests/Basic/OffloadArchTest.cpp - Test OffloadArch -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OffloadArch.h"
#include "gtest/gtest.h"

using namespace clang;

TEST(OffloadArchTest, basic) {
  EXPECT_TRUE(IsNVIDIAOffloadArch(OffloadArch::SM_20));
  EXPECT_TRUE(IsNVIDIAOffloadArch(OffloadArch::SM_120a));
  EXPECT_FALSE(IsNVIDIAOffloadArch(OffloadArch::GFX600));

  EXPECT_FALSE(IsAMDOffloadArch(OffloadArch::SM_120a));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::GFX600));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::GFX1201));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::GFX12_GENERIC));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::AMDGCNSPIRV));
  EXPECT_FALSE(IsAMDOffloadArch(OffloadArch::GRANITERAPIDS));

  EXPECT_TRUE(IsIntelOffloadArch(OffloadArch::GRANITERAPIDS));
  EXPECT_TRUE(IsIntelCPUOffloadArch(OffloadArch::GRANITERAPIDS));
  EXPECT_FALSE(IsIntelGPUOffloadArch(OffloadArch::GRANITERAPIDS));
  EXPECT_TRUE(IsIntelOffloadArch(OffloadArch::BMG_G21));
  EXPECT_FALSE(IsIntelCPUOffloadArch(OffloadArch::BMG_G21));
  EXPECT_TRUE(IsIntelGPUOffloadArch(OffloadArch::BMG_G21));

  EXPECT_FALSE(IsNVIDIAOffloadArch(OffloadArch::Generic));
  EXPECT_FALSE(IsAMDOffloadArch(OffloadArch::Generic));
  EXPECT_FALSE(IsIntelOffloadArch(OffloadArch::Generic));
}

TEST(OffloadArchTest, IntelGPUFamilyArchitectures) {
  EXPECT_TRUE(IsIntelGPUOffloadArch(OffloadArch::DG2));
  EXPECT_TRUE(IsIntelGPUOffloadArch(OffloadArch::MTL));
  EXPECT_TRUE(IsIntelGPUOffloadArch(OffloadArch::BMG));
  EXPECT_TRUE(IsIntelGPUOffloadArch(OffloadArch::PTL));
}

TEST(OffloadArchTest, IntelGPUFamilyArchParsing) {
  EXPECT_EQ(StringToOffloadArch("dg2"), OffloadArch::DG2);
  EXPECT_EQ(StringToOffloadArch("mtl"), OffloadArch::MTL);
  EXPECT_EQ(StringToOffloadArch("bmg"), OffloadArch::BMG);
  EXPECT_EQ(StringToOffloadArch("ptl"), OffloadArch::PTL);
}

TEST(OffloadArchTest, SYCLNVIDIAArchitectures) {
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_37));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_50));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_88));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_90a));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_100a));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_101f));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_103a));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_110f));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_120a));
  EXPECT_TRUE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::SM_121f));
  EXPECT_FALSE(IsSYCLSupportedNVidiaGPUArch(OffloadArch::GFX600));
}

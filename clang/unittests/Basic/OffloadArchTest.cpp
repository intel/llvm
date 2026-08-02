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

static OffloadArch parse(llvm::StringRef S) { return StringToOffloadArch(S); }

TEST(OffloadArchTest, VendorClassification) {
  EXPECT_TRUE(IsNVIDIAOffloadArch(parse("sm_20")));
  EXPECT_TRUE(IsNVIDIAOffloadArch(parse("sm_120a")));
  EXPECT_FALSE(IsNVIDIAOffloadArch(parse("gfx600")));

  EXPECT_FALSE(IsAMDOffloadArch(parse("sm_120a")));
  EXPECT_TRUE(IsAMDOffloadArch(parse("gfx600")));
  EXPECT_TRUE(IsAMDOffloadArch(parse("gfx1201")));
  EXPECT_TRUE(IsAMDOffloadArch(parse("gfx12-generic")));
  EXPECT_TRUE(IsAMDOffloadArch(parse("amdgcnspirv")));
  EXPECT_FALSE(IsAMDOffloadArch(parse("graniterapids")));

  EXPECT_TRUE(IsIntelOffloadArch(parse("graniterapids")));
  EXPECT_TRUE(IsIntelCPUOffloadArch(parse("graniterapids")));
  EXPECT_FALSE(IsIntelGPUOffloadArch(parse("graniterapids")));
  EXPECT_TRUE(IsIntelOffloadArch(parse("bmg_g21")));
  EXPECT_FALSE(IsIntelCPUOffloadArch(parse("bmg_g21")));
  EXPECT_TRUE(IsIntelGPUOffloadArch(parse("bmg_g21")));

  EXPECT_FALSE(IsNVIDIAOffloadArch(parse("generic")));
  EXPECT_FALSE(IsAMDOffloadArch(parse("generic")));
  EXPECT_FALSE(IsIntelOffloadArch(parse("generic")));
}

TEST(OffloadArchTest, Unknown) {
  EXPECT_TRUE(parse("not-a-real-arch").isUnknown());
  EXPECT_TRUE(parse("").isUnused());
}

// Names must round-trip through parse -> string.
TEST(OffloadArchTest, RoundTrip) {
  for (const char *Name :
       {"sm_52", "sm_90a", "gfx906", "gfx1201", "gfx12-generic", "amdgcnspirv",
        "graniterapids", "bmg_g21", "generic"}) {
    OffloadArch A = parse(Name);
    EXPECT_FALSE(A.isUnknown()) << Name;
    EXPECT_STREQ(OffloadArchToString(A), Name);
  }
}

TEST(OffloadArchTest, Defaults) {
  EXPECT_STREQ(OffloadArchToString(OffloadArch::CudaDefault()), "sm_75");
  EXPECT_STREQ(OffloadArchToString(OffloadArch::HIPDefault()), "gfx906");
}

TEST(OffloadArchTest, IntelGPUFamilyArchitectures) {
  EXPECT_TRUE(IsIntelGPUOffloadArch(parse("dg2")));
  EXPECT_TRUE(IsIntelGPUOffloadArch(parse("mtl")));
  EXPECT_TRUE(IsIntelGPUOffloadArch(parse("bmg")));
  EXPECT_TRUE(IsIntelGPUOffloadArch(parse("ptl")));
}

TEST(OffloadArchTest, IntelGPUFamilyArchParsing) {
  EXPECT_EQ(StringToOffloadArch("dg2"),
            OffloadArch::getIntel(OffloadArch::TargetArch::IntelGPU,
                                  OffloadArch::IntelArch::DG2));
  EXPECT_EQ(StringToOffloadArch("mtl"),
            OffloadArch::getIntel(OffloadArch::TargetArch::IntelGPU,
                                  OffloadArch::IntelArch::MTL));
  EXPECT_EQ(StringToOffloadArch("bmg"),
            OffloadArch::getIntel(OffloadArch::TargetArch::IntelGPU,
                                  OffloadArch::IntelArch::BMG));
  EXPECT_EQ(StringToOffloadArch("ptl"),
            OffloadArch::getIntel(OffloadArch::TargetArch::IntelGPU,
                                  OffloadArch::IntelArch::PTL));
}

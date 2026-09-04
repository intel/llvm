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

TEST(OffloadArchTest, TargetArchClassification) {
  OffloadArch NV = parse("sm_120a");
  EXPECT_TRUE(parse("sm_20").isNVPTX());
  EXPECT_TRUE(NV.isNVPTX());
  EXPECT_FALSE(NV.isAMDGPU());

  EXPECT_TRUE(parse("gfx600").isAMDGPU());
  EXPECT_TRUE(parse("gfx1201").isAMDGPU());
  EXPECT_TRUE(parse("gfx12-generic").isAMDGPU());
  EXPECT_FALSE(parse("gfx600").isNVPTX());

  OffloadArch SPIRV = parse("amdgcnspirv");
  EXPECT_FALSE(SPIRV.isAMDGPU());
  EXPECT_TRUE(SPIRV.isSPIRV());

  OffloadArch IntelCPU = parse("graniterapids");
  EXPECT_FALSE(IntelCPU.isAMDGPU());
  EXPECT_FALSE(IntelCPU.isSPIRV());
  EXPECT_TRUE(IntelCPU.isIntel());
  EXPECT_TRUE(IntelCPU.isIntelCPU());
  EXPECT_FALSE(IntelCPU.isIntelGPU());

  OffloadArch IntelGPU = parse("bmg_g21");
  EXPECT_TRUE(IntelGPU.isIntel());
  EXPECT_FALSE(IntelGPU.isIntelCPU());
  EXPECT_TRUE(IntelGPU.isIntelGPU());

  OffloadArch Generic = parse("generic");
  EXPECT_FALSE(Generic.isNVPTX());
  EXPECT_FALSE(Generic.isAMDGPU());
  EXPECT_FALSE(Generic.isSPIRV());
  EXPECT_FALSE(Generic.isIntel());
}

TEST(OffloadArchTest, Unknown) {
  EXPECT_TRUE(parse("not-a-real-arch").isUnknown());
  EXPECT_TRUE(parse("").isUnused());
}

// Names must round-trip through parse -> string.
TEST(OffloadArchTest, RoundTrip) {
  for (const char *Name :
       {"sm_52", "sm_90a", "gfx906", "gfx1201", "gfx12-generic", "amdgcnspirv",
        "graniterapids", "bmg_g21", "generic", "xe-bmg-g21", "xe-dg2",
        "igca_40r"}) {
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
  EXPECT_TRUE(parse("dg2").isIntelGPU());
  EXPECT_TRUE(parse("mtl").isIntelGPU());
  EXPECT_TRUE(parse("bmg").isIntelGPU());
  EXPECT_TRUE(parse("ptl").isIntelGPU());
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

// The names the GPU driver uses, as listed in IntelGPUArch.def: the name of an
// architecture, the IGCA level of a group of architectures, or the numeric form
// of a GMDID.
TEST(OffloadArchTest, IntelGPUArchNames) {
  EXPECT_TRUE(parse("xe-lnl-m").isIntelXeGPU());
  EXPECT_TRUE(parse("xe-lnl-m").isIntelGPU());
  EXPECT_TRUE(parse("xe-lnl-m").isIntel());
  EXPECT_TRUE(parse("xe-cri").isIntelXeGPU());
  EXPECT_TRUE(parse("igca_40r").isIntelXeGPU());
  EXPECT_TRUE(parse("xe_20.4.5").isIntelXeGPU());

  // A name that covers more than one release has no GMDID of its own, but is
  // still a name.
  EXPECT_TRUE(parse("xe-dg2").isIntelXeGPU());

  EXPECT_TRUE(parse("xe-lnl").isUnknown());
  EXPECT_TRUE(parse("igca_99").isUnknown());
}

// The revision component of a numeric name is not validated: every stepping of
// an architecture shares one name.
TEST(OffloadArchTest, IntelGPUNumericArchNames) {
  EXPECT_EQ(parse("xe_20.4.0"), parse("xe-lnl-m"));
  EXPECT_EQ(parse("xe_20.4.63"), parse("xe-lnl-m"));

  // Several architectures can share an architecture and a release, in which
  // case the first of them names the group.
  EXPECT_EQ(parse("xe_12.55.0"), parse("xe-acm-g10"));

  // An architecture and release that no entry has.
  EXPECT_TRUE(parse("xe_20.99.0").isUnknown());

  // The sentinel GMDID of a name that covers more than one release is not a
  // GMDID that a device reports.
  EXPECT_TRUE(parse("xe_0.0.0").isUnknown());

  for (const char *Name : {"xe_20.4", "xe_20.4.5.6", "xe_20.4.x", "xe_.4.5",
                           "xe_-20.4.5", "xe_20.4.-5", "xe_"})
    EXPECT_TRUE(parse(Name).isUnknown()) << Name;
}

// The IGCA level and the name of an architecture stay apart, as they name
// different sets of devices.
TEST(OffloadArchTest, IntelGPUArchNameAndIGCALevelDiffer) {
  EXPECT_NE(parse("igca_40r"), parse("xe-lnl-m"));
  EXPECT_STREQ(OffloadArchToString(parse("igca_40r")), "igca_40r");
  EXPECT_STREQ(OffloadArchToString(parse("xe-bmg-g31")), "xe-bmg-g31");
}

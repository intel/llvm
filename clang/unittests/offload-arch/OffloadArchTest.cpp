//===-- OffloadArchTest.cpp - Tests for offload-arch helpers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

// Defined in AMDGPUArchByHIP.cpp (non-static, compiled into this test).
#ifdef _WIN32
bool compareVersions(llvm::StringRef A, llvm::StringRef B);
llvm::SmallVector<std::string, 8> getCandidateBinPaths(llvm::StringRef ExeDir);
#endif

// Defined in LevelZeroArch.cpp (non-static, compiled into this test).
std::string getIntelGPUArchName(uint32_t IPVersion);

using namespace llvm;

cl::opt<bool> Verbose("offload-arch-test-verbose", cl::Hidden, cl::init(false));

#ifdef _WIN32

// --- compareVersions ---

TEST(CompareVersions, HigherVersionWins) {
  EXPECT_TRUE(
      compareVersions("C:/bin/amdhip64_7.dll", "C:/bin/amdhip64_6.dll"));
  EXPECT_FALSE(
      compareVersions("C:/bin/amdhip64_6.dll", "C:/bin/amdhip64_7.dll"));
}

TEST(CompareVersions, EqualVersionsReturnFalse) {
  EXPECT_FALSE(compareVersions("C:/a/amdhip64_7.dll", "C:/b/amdhip64_7.dll"));
}

TEST(CompareVersions, MultiDigitVersions) {
  EXPECT_TRUE(compareVersions("amdhip64_12.dll", "amdhip64_6.dll"));
}

TEST(CompareVersions, StableSortPreservesInsertionOrder) {
  std::vector<std::string> DLLs = {"C:/rocm/bin/amdhip64_7.dll",
                                   "C:/Windows/System32/amdhip64_7.dll"};
  llvm::stable_sort(DLLs, compareVersions);
  EXPECT_EQ(DLLs[0], "C:/rocm/bin/amdhip64_7.dll");
}

// --- getCandidateBinPaths ---

TEST(CandidateBinPaths, FindsParentBin) {
  auto Paths = getCandidateBinPaths("C:/root/lib/llvm/bin");
  bool Found = false;
  for (const auto &P : Paths)
    if (StringRef(P).equals_insensitive("C:/root/bin"))
      Found = true;
  EXPECT_TRUE(Found);
}

TEST(CandidateBinPaths, NoDuplicatesWhenExeInBin) {
  auto Paths = getCandidateBinPaths("C:/root/bin");
  int Count = 0;
  for (const auto &P : Paths)
    if (StringRef(P).equals_insensitive("C:/root/bin"))
      Count++;
  EXPECT_EQ(Count, 1);
}

TEST(CandidateBinPaths, CaseInsensitiveDedup) {
  // Paths differing only in case should not both appear.
  auto Paths = getCandidateBinPaths("C:/Root/Lib/Bin");
  int Count = 0;
  for (const auto &P : Paths)
    if (StringRef(P).equals_insensitive("C:/Root/bin"))
      Count++;
  EXPECT_LE(Count, 1);
}

TEST(CandidateBinPaths, StopsWithinBound) {
  auto Paths = getCandidateBinPaths("C:/a/b/c/d/e/f/g/h");
  // MaxParentLevels=6 + self = 7 max entries.
  EXPECT_LE(Paths.size(), 7u);
}

TEST(CandidateBinPaths, RootInput) {
  auto Paths = getCandidateBinPaths("C:/");
  // Should produce at least 1 entry (self) and not crash.
  EXPECT_GE(Paths.size(), 1u);
}

TEST(CandidateBinPaths, NonAsciiPath) {
  // Paths with non-ASCII characters should not crash.
  auto Paths = getCandidateBinPaths("C:/\xC3\xBCser/\xC3\xA4pp/bin");
  EXPECT_GE(Paths.size(), 1u);
}

TEST(CandidateBinPaths, UnicodePathDedup) {
  auto Paths =
      getCandidateBinPaths("C:/\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E/lib/bin");
  // Should produce entries without crashing on CJK characters.
  EXPECT_GE(Paths.size(), 1u);
}

TEST(CandidateBinPaths, NoDriveRootBin) {
  auto Paths = getCandidateBinPaths("C:\\Program Files\\AMD\\HIP\\bin");
  for (const auto &P : Paths)
    EXPECT_FALSE(StringRef(P).equals_insensitive("C:/bin"))
        << "Drive-root bin/ must not appear (DLL planting risk)";
}

#endif // _WIN32

// --- getIntelGPUArchName ---

namespace {
// Build a GMDID the way the Level Zero driver reports it.
constexpr uint32_t gmdid(uint32_t Architecture, uint32_t Release,
                         uint32_t Revision) {
  return (Architecture << 22) | (Release << 14) | Revision;
}
} // namespace

TEST(IntelGPUArchName, KnownArchitecturesGetAFriendlyName) {
  EXPECT_EQ(getIntelGPUArchName(gmdid(12, 60, 7)), "xe-pvc");
  EXPECT_EQ(getIntelGPUArchName(gmdid(20, 1, 4)), "xe-bmg-g21");
  EXPECT_EQ(getIntelGPUArchName(gmdid(35, 10, 0)), "xe-nvl-p");
  EXPECT_EQ(getIntelGPUArchName(gmdid(12, 0, 0)), "xe-tgllp");
}

// When several devices share an architecture and a release, the first one
// listed in IntelGPUArch.def names the whole group.
TEST(IntelGPUArchName, FirstNameOfAGroupWins) {
  EXPECT_EQ(getIntelGPUArchName(gmdid(30, 5, 0)), "xe-nvl-u");
  EXPECT_EQ(getIntelGPUArchName(gmdid(12, 55, 0)), "xe-acm-g10");
}

// The revision is not part of the lookup: every stepping of an architecture
// shares one name.
TEST(IntelGPUArchName, RevisionDoesNotAffectTheName) {
  EXPECT_EQ(getIntelGPUArchName(gmdid(12, 60, 0)), "xe-pvc");
  EXPECT_EQ(getIntelGPUArchName(gmdid(12, 60, 63)), "xe-pvc");
}

// An architecture that is not in the table still has to be named, so that a
// newer device is usable with a compiler that predates it.
TEST(IntelGPUArchName, UnknownArchitecturesGetANumericName) {
  EXPECT_EQ(getIntelGPUArchName(gmdid(40, 11, 0)), "xe_40.11.0");
  EXPECT_EQ(getIntelGPUArchName(gmdid(12, 99, 3)), "xe_12.99.3");
}

// Pre-Xe devices report a GMDID too, and none of them are in the table.
TEST(IntelGPUArchName, LegacyArchitecture) {
  EXPECT_EQ(getIntelGPUArchName(gmdid(9, 0, 9)), "xe_9.0.9");
}

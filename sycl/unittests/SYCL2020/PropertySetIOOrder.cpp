//==-- PropertySetIOOrder.cpp - property iteration order unit test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Regression test for CMPLRLLVM-77316: PropertySetRegistry::read must yield
// properties in insertion order. A hash container mismaps spec-const blob
// offsets on the SYCLBIN path, silently dropping set_specialization_constant.

#include <sycl/sycl.hpp>

#include <detail/property_set_io.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using namespace sycl::detail;

// Neither alphabetical nor libstdc++ hash order matches this sequence.
static constexpr const char *ExpectedOrder[] = {"beta", "alpha", "mid"};

static std::string makeBlob() {
  // Format: [<category>]\n<name>=<type>|<value>\n; type 1 == UINT32.
  std::ostringstream OS;
  OS << "[SYCL/specialization constants]\n";
  OS << "beta=1|10\n";
  OS << "alpha=1|20\n";
  OS << "mid=1|30\n";
  return OS.str();
}

// read() preserves the blob's property order.
TEST(PropertySetIOOrder, ReadPreservesInsertionOrder) {
  std::string Blob = makeBlob();
  auto Reg = PropertySetRegistry::read(Blob);
  ASSERT_NE(Reg, nullptr);

  auto SetIt = Reg->getPropSets().find("SYCL/specialization constants");
  ASSERT_NE(SetIt, Reg->getPropSets().end());

  std::vector<std::string> Names;
  for (const auto &Prop : SetIt->second)
    Names.push_back(Prop.first);

  ASSERT_EQ(Names.size(), 3u);
  EXPECT_EQ(Names[0], ExpectedOrder[0]);
  EXPECT_EQ(Names[1], ExpectedOrder[1]);
  EXPECT_EQ(Names[2], ExpectedOrder[2]);
}

// write() then read() preserves order.
TEST(PropertySetIOOrder, WriteReadRoundTripPreservesOrder) {
  std::string Blob = makeBlob();
  auto Reg = PropertySetRegistry::read(Blob);
  ASSERT_NE(Reg, nullptr);

  std::ostringstream OS;
  Reg->write(OS);
  std::string Serialized = OS.str();

  auto Reg2 = PropertySetRegistry::read(Serialized);
  ASSERT_NE(Reg2, nullptr);

  auto SetIt = Reg2->getPropSets().find("SYCL/specialization constants");
  ASSERT_NE(SetIt, Reg2->getPropSets().end());

  std::vector<std::string> Names;
  for (const auto &Prop : SetIt->second)
    Names.push_back(Prop.first);

  ASSERT_EQ(Names.size(), 3u);
  EXPECT_EQ(Names[0], ExpectedOrder[0]);
  EXPECT_EQ(Names[1], ExpectedOrder[1]);
  EXPECT_EQ(Names[2], ExpectedOrder[2]);
}

// operator[] appends new keys at the end.
TEST(PropertySetIOOrder, InsertionAppendsAtEnd) {
  PropertySetRegistry Reg;
  Reg.add("SYCL/specialization constants", "beta", uint32_t{10});
  Reg.add("SYCL/specialization constants", "alpha", uint32_t{20});
  Reg.add("SYCL/specialization constants", "mid", uint32_t{30});

  auto SetIt = Reg.getPropSets().find("SYCL/specialization constants");
  ASSERT_NE(SetIt, Reg.getPropSets().end());

  std::vector<std::string> Names;
  for (const auto &Prop : SetIt->second)
    Names.push_back(Prop.first);

  ASSERT_EQ(Names.size(), 3u);
  EXPECT_EQ(Names[0], ExpectedOrder[0]);
  EXPECT_EQ(Names[1], ExpectedOrder[1]);
  EXPECT_EQ(Names[2], ExpectedOrder[2]);
}

//===- llvm/unittest/Support/PropertySetIO.cpp - Property set I/O tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::util;

namespace {

TEST(PropertySet, IntValuesIO) {
  // '1' in '1|20' means 'integer property'
  auto Content = "[Staff/Ages]\n"
                 "person1=1|20\n"
                 "person2=1|25\n"
                 "[Staff/Experience]\n"
                 "person1=1|1\n"
                 "person2=1|2\n"
                 "person3=1|12\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = PropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "PropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  ASSERT_EQ(Serialized, Content);
}

TEST(PropertySet, ByteArrayValuesIO) {
  // '2' in '2|...' means 'byte array property', Base64-encoded
  // encodes the following byte arrays:
  //   { 8, 0, 0, 0, 0, 0, 0, 0, 0x1 };
  //   { 40, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0x7F, 0xFF, 0x70 };
  // first 8 bytes are the size in bits (40) of what follows (5 bytes).

  auto Content = "[Opt/Param]\n"
                 "kernel1=2|IAAAAAAAAAQA\n"
                 "kernel2=2|oAAAAAAAAAw///3/wB\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = PropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "PropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  ASSERT_EQ(Serialized, Content);
}

TEST(PropertySet, EmptyByteArrayValue) {
  // An empty container has a null data pointer, which must not be passed to
  // memcpy.
  PropertyValue Prop{std::vector<char>{}};
  ASSERT_EQ(Prop.getType(), PropertyValue::BYTE_ARRAY);
  ASSERT_EQ(Prop.getByteArraySize(), 0u);
}

// Builds a registry with a single set holding the given key/value pairs.
static PropertySetRegistry
makeRegistry(StringRef SetName,
             ArrayRef<std::pair<StringRef, uint32_t>> Props) {
  PropertySetRegistry Reg;
  for (const auto &[Name, Val] : Props)
    Reg.add(SetName, Name, Val);
  return Reg;
}

static std::string serialize(const PropertySetRegistry &Reg) {
  std::string S;
  raw_string_ostream OS(S);
  Reg.write(OS);
  return S;
}

TEST(PropertySetFold, PerTargetSetName) {
  EXPECT_EQ(getPerTargetSetName("igca_10", "SYCL/device requirements"),
            "SYCL/igca_10/device requirements");
  // A name without a namespace component still gets a target.
  EXPECT_EQ(getPerTargetSetName("igca_10", "flat"), "igca_10/flat");
}

TEST(PropertySetFold, UniformPropertiesStayFlat) {
  auto A = makeRegistry("SYCL/misc properties", {{"optLevel", 2}});
  auto B = makeRegistry("SYCL/misc properties", {{"optLevel", 2}});
  PerTargetPropertySets Inputs[] = {{"igca_10", &A}, {"igca_20", &B}};
  // Both targets agree, so the output is what a single target would produce.
  EXPECT_EQ(serialize(foldPerTargetProperties(Inputs)), serialize(A));
}

TEST(PropertySetFold, DivergingPropertiesGoPerTarget) {
  auto A = makeRegistry("SYCL/misc properties", {{"optLevel", 2}});
  auto B = makeRegistry("SYCL/misc properties", {{"optLevel", 3}});
  PerTargetPropertySets Inputs[] = {{"igca_10", &A}, {"igca_20", &B}};
  EXPECT_EQ(serialize(foldPerTargetProperties(Inputs)),
            "[SYCL/igca_10/misc properties]\n"
            "optLevel=1|2\n"
            "[SYCL/igca_20/misc properties]\n"
            "optLevel=1|3\n");
}

TEST(PropertySetFold, UniformAndDivergingPropertiesSplitApart) {
  auto A = makeRegistry("SYCL/misc properties",
                        {{"optLevel", 2}, {"isEsimdImage", 1}});
  auto B = makeRegistry("SYCL/misc properties",
                        {{"optLevel", 3}, {"isEsimdImage", 1}});
  PerTargetPropertySets Inputs[] = {{"igca_10", &A}, {"igca_20", &B}};
  // optLevel differs so it is stored per target; isEsimdImage agrees so it
  // stays in the flat set.
  EXPECT_EQ(serialize(foldPerTargetProperties(Inputs)),
            "[SYCL/igca_10/misc properties]\n"
            "optLevel=1|2\n"
            "[SYCL/igca_20/misc properties]\n"
            "optLevel=1|3\n"
            "[SYCL/misc properties]\n"
            "isEsimdImage=1|1\n");
}

TEST(PropertySetFold, MissingForOneTargetIsNotUniform) {
  auto A = makeRegistry("SYCL/misc properties", {{"optLevel", 2}});
  PropertySetRegistry B; // no properties at all for this target
  PerTargetPropertySets Inputs[] = {{"igca_10", &A}, {"igca_20", &B}};
  // "Absent" is meaningful, so it must not be flattened into a value that
  // would then apply to igca_20 as well.
  EXPECT_EQ(serialize(foldPerTargetProperties(Inputs)),
            "[SYCL/igca_10/misc properties]\n"
            "optLevel=1|2\n");
}

TEST(PropertySetFold, SingleTargetIsUnchanged) {
  auto A = makeRegistry("SYCL/misc properties", {{"optLevel", 2}});
  PerTargetPropertySets Inputs[] = {{"igca_10", &A}};
  EXPECT_EQ(serialize(foldPerTargetProperties(Inputs)), serialize(A));
}

TEST(PropertySetFold, NoInputsGivesEmptyRegistry) {
  EXPECT_EQ(serialize(foldPerTargetProperties({})), "");
}
} // namespace

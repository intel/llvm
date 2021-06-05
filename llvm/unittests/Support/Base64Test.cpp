//===- llvm/unittest/Support/Base64Test.cpp - Base64 tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the Base64 functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Base64.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
/// Tests an arbitrary set of bytes passed as \p Input.
void TestBase64(StringRef Input, StringRef Final) {
  auto Res = encodeBase64(Input);
  EXPECT_EQ(Res, Final);
}

} // namespace

TEST(Base64Test, Base64) {
  // from: https://tools.ietf.org/html/rfc4648#section-10
  TestBase64("", "");
  TestBase64("f", "Zg==");
  TestBase64("fo", "Zm8=");
  TestBase64("foo", "Zm9v");
  TestBase64("foob", "Zm9vYg==");
  TestBase64("fooba", "Zm9vYmE=");
  TestBase64("foobar", "Zm9vYmFy");

  // With non-printable values.
  char NonPrintableVector[] = {0x00, 0x00, 0x00,       0x46,
                               0x00, 0x08, (char)0xff, (char)0xee};
  TestBase64({NonPrintableVector, sizeof(NonPrintableVector)}, "AAAARgAI/+4=");

  // Large test case
  char LargeVector[] = {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b,
                        0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f,
                        0x78, 0x20, 0x6a, 0x75, 0x6d, 0x70, 0x73, 0x20, 0x6f,
                        0x76, 0x65, 0x72, 0x20, 0x31, 0x33, 0x20, 0x6c, 0x61,
                        0x7a, 0x79, 0x20, 0x64, 0x6f, 0x67, 0x73, 0x2e};
  TestBase64({LargeVector, sizeof(LargeVector)},
             "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIDEzIGxhenkgZG9ncy4=");
}

TEST(Base64Test, RoundTrip) {
  using byte = unsigned char;
  const byte Arr0[] = {0x1};
  const byte Arr1[] = {0x81}; // 0x81 - highest and lowest bits are set
  const byte Arr2[] = {0x81, 0x81};
  const byte Arr3[] = {0x81, 0x81, 0x81};
  const byte Arr4[] = {0x81, 0x81, 0x81, 0x81};
  const byte Arr5[] = {0xFF, 0xFF, 0x7F, 0xFF, 0x70};
  const byte Arr6[] = {40, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0x7F, 0xFF, 0x70};
  const byte Arr7[] = {8, 0, 0, 0, 0, 0, 0, 0, 0x1};
  // 2 tests below model real usage case for argument opt info propagation:
  // { 0x7E, 0x06 }
  // 11001111110 - 11 arguments total, 3 remain
  // encoded: "LAAAAAAAAAgfGA"
  const byte Arr8[] = {11, 0, 0, 0, 0, 0, 0, 0, 0x7E, 0x06};
  // 0x01
  // 01 - 2 arguments total, 1 remains
  // encoded: "CAAAAAAAAAQA"
  const byte Arr9[] = {2, 0, 0, 0, 0, 0, 0, 0, 0x1};

  struct {
    const byte *Ptr;
    size_t Size;
  } Tests[] = {{Arr0, sizeof(Arr0)}, {Arr1, sizeof(Arr1)}, {Arr2, sizeof(Arr2)},
               {Arr3, sizeof(Arr3)}, {Arr4, sizeof(Arr4)}, {Arr5, sizeof(Arr5)},
               {Arr6, sizeof(Arr6)}, {Arr7, sizeof(Arr7)}, {Arr8, sizeof(Arr8)},
               {Arr9, sizeof(Arr9)}};

  for (size_t I = 0; I < sizeof(Tests) / sizeof(Tests[0]); ++I) {
    std::string Encoded;
    size_t Len;
    {
      llvm::raw_string_ostream OS(Encoded);
      Len = Base64::encode(Tests[I].Ptr, OS, Tests[I].Size);
    }
    if (Len != Encoded.size()) {
      FAIL() << "Base64::encode failed on test " << I << "\n";
      continue;
    }
    std::unique_ptr<byte> Decoded(new byte[Base64::getDecodedSize(Len)]);
    Expected<size_t> Res = Base64::decode(Encoded.data(), Decoded.get(), Len);

    if (!Res) {
      FAIL() << "Base64::decode failed on test " << I << "\n";
      continue;
    }
    if (Res.get() != Tests[I].Size) {
      FAIL() << "Base64::decode length mismatch, test " << I << "\n";
      continue;
    }
    std::string Gold((const char *)Tests[I].Ptr, Tests[I].Size);
    std::string Test((const char *)Decoded.get(), Res.get());

    if (Gold != Test) {
      FAIL() << "Base64::decode result mismatch, test " << I << "\n";
    }
  }
}

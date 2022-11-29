//==------------ bfloat16_host.cpp - SYCL vectors test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

// Helper to convert the expected bits to float value to compare with the result
typedef union {
  float Value;
  struct {
    uint32_t Mantissa : 23;
    uint32_t Exponent : 8;
    uint32_t Sign : 1;
  } RawData;
} floatConvHelper;

float bitsToFloatConv(std::string Bits) {
  floatConvHelper Helper;
  Helper.RawData.Sign = static_cast<uint32_t>(Bits[0] - '0');
  uint32_t Exponent = 0;
  for (size_t I = 1; I != 9; ++I)
    Exponent = Exponent + static_cast<uint32_t>(Bits[I] - '0') * pow(2, 8 - I);
  Helper.RawData.Exponent = Exponent;
  uint32_t Mantissa = 0;
  for (size_t I = 9; I != 32; ++I)
    Mantissa = Mantissa + static_cast<uint32_t>(Bits[I] - '0') * pow(2, 31 - I);
  Helper.RawData.Mantissa = Mantissa;
  return Helper.Value;
}

bool check_bf16_from_float(float Val, uint16_t Expected) {
  sycl::ext::oneapi::bfloat16 B = Val;
  uint16_t Result = *reinterpret_cast<uint16_t *>(&B);
  if (Result != Expected) {
    std::cout << "from_float check for Val = " << Val << " failed!\n"
              << "Expected " << Expected << " Got " << Result << "\n";
    return false;
  }
  return true;
}

bool check_bf16_to_float(uint16_t Val, float Expected) {
  float Result = *reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(&Val);
  if (Result != Expected) {
    std::cout << "to_float check for Val = " << Val << " failed!\n"
              << "Expected " << Expected << " Got " << Result << "\n";
    return false;
  }
  return true;
}

int main() {
  bool Success =
      check_bf16_from_float(0.0f, std::stoi("0000000000000000", nullptr, 2));
  Success &=
      check_bf16_from_float(42.0f, std::stoi("100001000101000", nullptr, 2));
  Success &= check_bf16_from_float(std::numeric_limits<float>::min(),
                                   std::stoi("0000000010000000", nullptr, 2));
  Success &= check_bf16_from_float(std::numeric_limits<float>::max(),
                                   std::stoi("0111111110000000", nullptr, 2));
  Success &= check_bf16_from_float(std::numeric_limits<float>::quiet_NaN(),
                                   std::stoi("1111111111000001", nullptr, 2));

  // see https://float.exposed/b0xffff
  Success &= check_bf16_to_float(
      0, bitsToFloatConv(std::string("00000000000000000000000000000000")));
  Success &= check_bf16_to_float(
      1, bitsToFloatConv(std::string("00000000000000010000000000000000")));
  Success &= check_bf16_to_float(
      42, bitsToFloatConv(std::string("00000000001010100000000000000000")));
  Success &= check_bf16_to_float(
      // std::numeric_limits<uint16_t>::max() - 0xffff is bfloat16 -Nan and
      // -Nan == -Nan check in check_bf16_to_float would fail, so use not Nan:
      65407, bitsToFloatConv(std::string("11111111011111110000000000000000")));
  if (!Success)
    return -1;
  return 0;
}

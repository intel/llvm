//==------------ bfloat16_host.cpp - SYCL vectors test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %RUN_ON_HOST %t.out
#include <sycl/ext/intel/experimental/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using sycl::ext::intel::experimental::bfloat16;

// Helper to convert the expected bits to float value to compare with the result
typedef union {
  float Value;
  struct {
    uint32_t Mantissa : 23;
    uint32_t Exponent : 8;
    uint32_t Sign : 1;
  } RawData;
} floatConvHelper;

float bistToFloatConv(std::string &Bits) {
  floatConvHelper &Helper;
  Helper.RawData.Sign = static_cast<uint32_t>(Bits[0] - '0');
  uint32_t Exponent = 0;
  for (size_t I = 1; I != 9; ++I)
    Exponent = Exponent + static_cast<uint32_t>(Bits[I] - '0') * pow(2, 8 - I);
  Helper.RawData.Exponent = Exponent;
  uint32_t Mantissa = 0;
  for (size_t I = 9; I != 32; ++I)
    Mantissa = Mantissa + static_cast<uint32_t>(Bits[I] - '0') * pow(2, 8 - I);
  Helper.RawData.Mantissa = Mantissa;
}

inline bool check_bf16_from_float(float &Val, uint16_t &Expected) {
  if (from_float(Val) != Expected) {
    std::cout << "from_float check for Val = " << Val << " failed!\n";
    return false;
  }
  return true;
}

inline bool check_bf16_to_float(uint16_t &Val, float &Expected) {
  if (to_float(Val) != Expected) {
    std::cout << "to_float check for Val = " << Val << " failed!\n";
    return false;
  }
  return true;
}

int main() {
  bool Success =
      check_bf16_from_float(0.0f, std::stoi("0000000000000000", nullptr, 2));
  Success &= check_bf16_from_float(42.0f,
                                   std::stoi("100001000101000", nullptr, 2));
  Success &= check_bf16_from_float(std::numeric_limits<float>::min(),
                                   std::stoi("0000000010000000", nullptr, 2));
  Success &= check_bf16_from_float(std::numeric_limits<float>::max(),
                                   std::stoi("0111111110000000", nullptr, 2));
  Success &= check_bf16_from_float(std::numeric_limits<float>::quiet_NaN(),
                                   std::stoi("1111111111000001", nullptr, 2));

  Success &=
      check_bf16_to_float(to_float(0),
                          bitToFloatConv("00000000000000000000000000000000"));
  Success &=
      check_bf16_to_float(to_float(1),
                          bitToFloatConv("01000111100000000000000000000000"));
  Success &=
      check_bf16_to_float(to_float(42),
                          bitToFloatConv("00000000001010100000000000000000"));
  Success &=
      check_bf16_to_float(to_float(std::numeric_limits<uint16_t>::max()),
                          bitToFloatConv("11111111111111110000000000000000"));
  if (!Success)
    return -1;
  return 0;
}

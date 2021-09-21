//==------- pi_arguments_handler.cpp --- A test for XPTI PI args helper ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "pi_arguments_handler.hpp"

#include <CL/sycl/detail/pi.h>

#include <array>

TEST(PiArgumentsHandlerTest, CanUnpackArguments) {
  sycl::xpti_helpers::PiArgumentsHandler Handler;

  const pi_uint32 NumPlatforms = 42;
  pi_platform *Platforms = new pi_platform[NumPlatforms];

  Handler.set_piPlatformsGet([&](const pi_plugin &, std::optional<pi_result>,
                                 pi_uint32 NP, pi_platform *Plts,
                                 pi_uint32 *Ret) {
    EXPECT_EQ(NP, NumPlatforms);
    EXPECT_EQ(Platforms, Plts);
    EXPECT_EQ(Ret, nullptr);
  });

  constexpr size_t Size = sizeof(pi_uint32) + 2 * sizeof(void *);
  std::array<unsigned char, Size> Data{0};
  *reinterpret_cast<pi_uint32 *>(Data.data()) = NumPlatforms;
  *reinterpret_cast<pi_platform **>(Data.data() + sizeof(pi_uint32)) =
      Platforms;

  pi_plugin Plugin{};
  uint32_t ID = static_cast<uint32_t>(sycl::detail::PiApiKind::piPlatformsGet);
  Handler.handle(ID, Plugin, std::nullopt, Data.data());

  delete[] Platforms;
}

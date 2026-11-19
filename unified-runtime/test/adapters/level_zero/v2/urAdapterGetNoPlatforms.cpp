// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check that the L0v2 adapter's urAdapterGet reports zero adapters with
// UR_RESULT_SUCCESS (rather than an error) when no Level Zero platforms are
// available.
//
// The adapter entry point is called directly rather than through the UR
// loader: when more than one adapter is loaded the loader intercepts
// urAdapterGet and discards per-adapter errors, which would hide the result.
//
// Pointing the L0 loader at a nonexistent driver library guarantees that no
// L0 drivers, and therefore no platforms, are found.
//
// RUN: env ZE_ENABLE_ALT_DRIVERS=/nonexistent/libze_fake_driver.so ./adapter_get_no_platforms-test
// REQUIRES: v2

#include "level_zero/v2/ur_interface_loader.hpp"

#include <gtest/gtest.h>

namespace v2 = ur::level_zero::v2;

// Only a single urAdapterGet call is made per process: with no platforms the
// adapter tears down the statically linked L0 loader context.
TEST(urAdapterGetNoPlatforms, ReturnsSuccessAndZeroAdapters) {
  ur_adapter_handle_t Adapter = nullptr;
  uint32_t NumAdapters = 42;
  ASSERT_EQ(v2::urAdapterGet(1, &Adapter, &NumAdapters), UR_RESULT_SUCCESS);
  EXPECT_EQ(NumAdapters, 0u);
  EXPECT_EQ(Adapter, nullptr);
}

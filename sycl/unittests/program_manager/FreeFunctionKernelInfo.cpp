//==------------------- FreeFunctionKernelInfo.cpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the free-function kernel global-info registration path
// (sycl::detail::free_function_info_map::add / remove), which is driven by the
// compiler-emitted GlobalMapUpdater object in the integration header.
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/kernel_global_info.hpp>

#include <optional>

#include <gtest/gtest.h>

using namespace sycl;

// While the runtime is alive, add() then remove() must round-trip through the
// ProgramManager's free-function global-info map.
TEST(FreeFunctionKernelInfo, AddRemoveWhileRuntimeAlive) {
  ASSERT_TRUE(detail::GlobalHandler::isInstanceAlive());

  static const char *const Names[] = {"ffk_alive_kernel_a",
                                      "ffk_alive_kernel_b"};
  static const unsigned Sizes[] = {3u, 7u};
  constexpr unsigned Count = 2;

  auto &PM = detail::ProgramManager::getInstance();

  detail::free_function_info_map::add(Names, Sizes, Count);
  EXPECT_EQ(PM.getKernelGlobalInfoDesc(Names[0]), std::optional<unsigned>(3u));
  EXPECT_EQ(PM.getKernelGlobalInfoDesc(Names[1]), std::optional<unsigned>(7u));

  detail::free_function_info_map::remove(Names, Sizes, Count);
  EXPECT_FALSE(PM.getKernelGlobalInfoDesc(Names[0]).has_value());
  EXPECT_FALSE(PM.getKernelGlobalInfoDesc(Names[1]).has_value());
}

// add()/remove() must be a safe no-op (not a use-after-free) when the
// GlobalMapUpdater destructor runs after the runtime is torn down.
TEST(FreeFunctionKernelInfo, AddRemoveAfterRuntimeTeardownIsNoop) {
  static const char *const Names[] = {"ffk_teardown_kernel_a",
                                      "ffk_teardown_kernel_b"};
  static const unsigned Sizes[] = {5u, 11u};
  constexpr unsigned Count = 2;

  // Simulate the runtime having been torn down.
  detail::GlobalHandler *Saved = detail::GlobalHandler::detachGlobalHandler();
  ASSERT_FALSE(detail::GlobalHandler::isInstanceAlive());

  // Must not touch the detached singleton (no crash / no use-after-free).
  detail::free_function_info_map::remove(Names, Sizes, Count);
  detail::free_function_info_map::add(Names, Sizes, Count);

  // Restore the runtime and confirm nothing was registered while it was dead.
  detail::GlobalHandler::restoreGlobalHandler(Saved);
  ASSERT_TRUE(detail::GlobalHandler::isInstanceAlive());

  auto &PM = detail::ProgramManager::getInstance();
  EXPECT_FALSE(PM.getKernelGlobalInfoDesc(Names[0]).has_value());
  EXPECT_FALSE(PM.getKernelGlobalInfoDesc(Names[1]).has_value());
}

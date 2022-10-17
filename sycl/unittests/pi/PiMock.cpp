//==--------- PiMock.cpp --- A test for mock helper API's ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>

#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

using namespace sycl;

pi_result piProgramBuildRedefine(pi_program, pi_uint32, const pi_device *,
                                 const char *, void (*)(pi_program, void *),
                                 void *) {
  return PI_ERROR_INVALID_BINARY;
}

pi_result piKernelCreateRedefine(pi_program, const char *, pi_kernel *) {
  return PI_ERROR_INVALID_DEVICE;
}

TEST(PiMockTest, ConstructFromQueue) {
  sycl::unittest::PiMock Mock;
  queue MockQ{Mock.getPlatform().get_devices()[0]};
  queue NormalQ;
  if (NormalQ.is_host()) {
    std::cerr << "Not run due to host-only environment\n";
    return;
  }

  const auto &NormalPiPlugin =
      detail::getSyclObjImpl(NormalQ)->getPlugin().getPiPlugin();
  const auto &MockedQueuePiPlugin =
      detail::getSyclObjImpl(MockQ)->getPlugin().getPiPlugin();
  const auto &PiMockPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin().getPiPlugin();
  EXPECT_EQ(&MockedQueuePiPlugin, &PiMockPlugin)
      << "The mocked object and the PiMock instance must share the same plugin";
  EXPECT_EQ(&NormalPiPlugin, &MockedQueuePiPlugin)
      << "Normal and mock platforms must share the same plugin";
}

TEST(PiMockTest, ConstructFromPlatform) {
  sycl::unittest::PiMock Mock;
  sycl::platform MockPlatform = Mock.getPlatform();
  platform NormalPlatform(default_selector{});

  const auto &NormalPiPlugin =
      detail::getSyclObjImpl(NormalPlatform)->getPlugin().getPiPlugin();
  const auto &MockedPlatformPiPlugin =
      detail::getSyclObjImpl(MockPlatform)->getPlugin().getPiPlugin();
  const auto &PiMockPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin().getPiPlugin();
  EXPECT_EQ(&MockedPlatformPiPlugin, &PiMockPlugin)
      << "The mocked object and the PiMock instance must share the same plugin";
  EXPECT_EQ(&NormalPiPlugin, &MockedPlatformPiPlugin)
      << "Normal and mock platforms must share the same plugin";
}

TEST(PiMockTest, RedefineAPI) {
  sycl::unittest::PiMock Mock;
  const auto &MockPiPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin().getPiPlugin();
  const auto &Table = MockPiPlugin.PiFunctionTable;

  // Pass a function pointer
  Mock.redefine<detail::PiApiKind::piProgramBuild>(piProgramBuildRedefine);
  EXPECT_EQ(Table.piProgramBuild, &piProgramBuildRedefine)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a std::function
  Mock.redefine<detail::PiApiKind::piKernelCreate>({piKernelCreateRedefine});
  EXPECT_EQ(Table.piKernelCreate, &piKernelCreateRedefine)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a captureless lambda
  auto *OldFuncPtr = Table.piProgramRetain;
  auto Lambda = [](pi_program) -> pi_result {
    return PI_ERROR_INVALID_PROGRAM;
  };
  EXPECT_FALSE(OldFuncPtr == Lambda)
      << "Lambda is the same as the existing function.";
  Mock.redefine<detail::PiApiKind::piProgramRetain>(Lambda);
  EXPECT_FALSE(Table.piProgramRetain == OldFuncPtr)
      << "Passing a lambda didn't change the function table entry";
  ASSERT_FALSE(Table.piProgramRetain == nullptr)
      << "Passing a lambda set the table entry to a null pointer";
}

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

static bool GpiProgramBuildRedefineCalled = false;
static bool GpiKernelCreateRedefineCalled = false;
static bool GpiProgramRetainCalled = false;
static bool GpiContextCreateRedefineCalledAfter = false;
static bool GpiQueueCreateRedefineCalledBefore = false;

pi_result piQueueCreateRedefineBefore(pi_context context, pi_device device,
                                      pi_queue_properties properties,
                                      pi_queue *queue) {
  // The context should have been set by the original function
  GpiQueueCreateRedefineCalledBefore = *queue == nullptr;
  // Returning an error should stop calls to all redefined functions
  return PI_ERROR_INVALID_OPERATION;
}

pi_result piContextCreateRedefineAfter(
    const pi_context_properties *properties, pi_uint32 num_devices,
    const pi_device *devices,
    void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb,
                       void *user_data),
    void *user_data, pi_context *ret_context) {
  // The context should have been set by the original function
  GpiContextCreateRedefineCalledAfter = *ret_context != nullptr;
  return PI_SUCCESS;
}

pi_result piProgramBuildRedefine(pi_program, pi_uint32, const pi_device *,
                                 const char *, void (*)(pi_program, void *),
                                 void *) {
  GpiProgramBuildRedefineCalled = true;
  return PI_SUCCESS;
}

pi_result piKernelCreateRedefine(pi_program, const char *, pi_kernel *) {
  GpiKernelCreateRedefineCalled = true;
  return PI_SUCCESS;
}

TEST(PiMockTest, ConstructFromQueue) {
  sycl::unittest::PiMock Mock;
  queue MockQ{Mock.getPlatform().get_devices()[0]};
  queue NormalQ;

  const auto &NormalPiPlugin =
      detail::getSyclObjImpl(NormalQ)->getPlugin()->getPiPlugin();
  const auto &MockedQueuePiPlugin =
      detail::getSyclObjImpl(MockQ)->getPlugin()->getPiPlugin();
  const auto &PiMockPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin()->getPiPlugin();
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
      detail::getSyclObjImpl(NormalPlatform)->getPlugin()->getPiPlugin();
  const auto &MockedPlatformPiPlugin =
      detail::getSyclObjImpl(MockPlatform)->getPlugin()->getPiPlugin();
  const auto &PiMockPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin()->getPiPlugin();
  EXPECT_EQ(&MockedPlatformPiPlugin, &PiMockPlugin)
      << "The mocked object and the PiMock instance must share the same plugin";
  EXPECT_EQ(&NormalPiPlugin, &MockedPlatformPiPlugin)
      << "Normal and mock platforms must share the same plugin";
}

TEST(PiMockTest, RedefineAPI) {
  sycl::unittest::PiMock Mock;
  const auto &MockPiPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin()->getPiPlugin();
  const auto &Table = MockPiPlugin.PiFunctionTable;

  // Pass a function pointer
  Mock.redefine<detail::PiApiKind::piProgramBuild>(piProgramBuildRedefine);
  Table.piProgramBuild(/*pi_program*/ nullptr, /*num_devices=*/0,
                       /*device_list = */ nullptr,
                       /*options=*/nullptr, /*pfn_notify=*/nullptr,
                       /*user_data=*/nullptr);

  EXPECT_TRUE(GpiProgramBuildRedefineCalled)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a std::function
  Mock.redefine<detail::PiApiKind::piKernelCreate>({piKernelCreateRedefine});

  Table.piKernelCreate(/*pi_program=*/nullptr, /*kernel_name=*/nullptr,
                       /*pi_kernel=*/nullptr);
  EXPECT_TRUE(GpiKernelCreateRedefineCalled)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a captureless lambda
  auto Lambda = [](pi_program) -> pi_result {
    GpiProgramRetainCalled = true;
    return PI_SUCCESS;
  };
  Mock.redefine<detail::PiApiKind::piProgramRetain>(Lambda);
  Table.piProgramRetain(/*pi_program=*/nullptr);

  EXPECT_TRUE(GpiProgramRetainCalled)
      << "Passing a lambda didn't change the function table entry";
}

TEST(PiMockTest, RedefineAfterAPI) {
  sycl::unittest::PiMock Mock;

  const auto &MockPiPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin()->getPiPlugin();
  const auto &Table = MockPiPlugin.PiFunctionTable;

  // Pass a function pointer
  Mock.redefineAfter<detail::PiApiKind::piContextCreate>(
      piContextCreateRedefineAfter);

  pi_context PIContext = nullptr;
  Table.piContextCreate(
      /*pi_context_properties=*/nullptr, /*num_devices=*/0,
      /*devices=*/nullptr, /*pfn_notify=*/nullptr,
      /*user_data=*/nullptr, &PIContext);

  EXPECT_TRUE(GpiContextCreateRedefineCalledAfter)
      << "The additional function is not called after the original one";
}

TEST(PiMockTest, RedefineBeforeAPI) {
  sycl::unittest::PiMock Mock;

  const auto &MockPiPlugin =
      detail::getSyclObjImpl(Mock.getPlatform())->getPlugin()->getPiPlugin();
  const auto &Table = MockPiPlugin.PiFunctionTable;

  // Pass a function pointer
  Mock.redefineBefore<detail::PiApiKind::piQueueCreate>(
      piQueueCreateRedefineBefore);

  pi_queue Queue = nullptr;
  Table.piQueueCreate(/*pi_context=*/nullptr, /*pi_device=*/nullptr,
                      /*pi_queue_properties=*/0, &Queue);

  EXPECT_TRUE(GpiQueueCreateRedefineCalledBefore)
      << "The additional function is not called before the original one";

  EXPECT_TRUE(nullptr == Queue) << "Queue is expected to be non-initialized as "
                                   "the original function should not be called";
}

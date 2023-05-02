//==---- test_ext_command_buffers.cpp --- PI unit tests
//---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "spirv-source.hpp"

#include "gtest/gtest.h"

#include "TestGetPlugin.hpp"
#include <detail/plugin.hpp>
#include <pi_level_zero.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

class LevelZeroCommandBuffersTest : public ::testing::Test {
protected:
  std::optional<detail::plugin> Plugin =
      pi::initializeAndGet(backend::ext_oneapi_level_zero);
  pi_context Context = nullptr;
  pi_device Device = nullptr;
  pi_queue Queue = nullptr;

  void SetUp() override {
    // skip the tests if the Level Zero backend is not available
    if (!Plugin.has_value()) {
      GTEST_SKIP();
    }

    pi_uint32 NumPlatforms = 0;
    pi_platform Platform = nullptr;

    ASSERT_EQ(Plugin->getBackend(), backend::level_zero);

    ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &NumPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  NumPlatforms, &Platform, nullptr)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_GE(NumPlatforms, 1u);
    ASSERT_NE(Platform, nullptr);

    ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piDevicesGet>(
                  Platform, PI_DEVICE_TYPE_GPU, 1, &Device, nullptr)),
              PI_SUCCESS)
        << "piDevicesGet failed.\n";

    ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piContextCreate>(
                  nullptr, 1, &Device, nullptr, nullptr, &Context)),
              PI_SUCCESS)
        << "piContextCreate failed.\n";

    ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piQueueCreate>(
                  Context, Device, 0, &Queue)),
              PI_SUCCESS)
        << "piContextCreate failed.\n";

    EXPECT_NE(Context, nullptr);
  }

  void TearDown() override {
    if (Plugin.has_value()) {
      Plugin->call<detail::PiApiKind::piQueueRelease>(Queue);
      Plugin->call<detail::PiApiKind::piDeviceRelease>(Device);
      Plugin->call<detail::PiApiKind::piContextRelease>(Context);
    }
  }

  LevelZeroCommandBuffersTest() = default;

  ~LevelZeroCommandBuffersTest() = default;
};

TEST_F(LevelZeroCommandBuffersTest, piextCommandBufferCreate) {
  // Create command-buffer
  pi_ext_command_buffer CommandBuffer = nullptr;
  pi_ext_command_buffer_desc CommandBufferDesc = {
      PI_EXT_STRUCTURE_TYPE_COMMAND_BUFFER_DESC, nullptr, nullptr};
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferCreate>(
                Context, Device, &CommandBufferDesc, &CommandBuffer)),
            PI_SUCCESS);

  ASSERT_NE(CommandBuffer, nullptr);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferRelease>(
                CommandBuffer)),
            PI_SUCCESS);
}

TEST_F(LevelZeroCommandBuffersTest, piextCommandBufferReleaseRetain) {
  // Create command-buffer
  pi_ext_command_buffer CommandBuffer = nullptr;
  pi_ext_command_buffer_desc CommandBufferDesc;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferCreate>(
                Context, Device, &CommandBufferDesc, &CommandBuffer)),
            PI_SUCCESS);
  ASSERT_EQ(CommandBuffer->RefCount.load(), 1u);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferRetain>(
                CommandBuffer)),
            PI_SUCCESS);
  ASSERT_EQ(CommandBuffer->RefCount.load(), 2u);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferRelease>(
                CommandBuffer)),
            PI_SUCCESS);
  ASSERT_EQ(CommandBuffer->RefCount.load(), 1u);
}

TEST_F(LevelZeroCommandBuffersTest, piextCommandBufferFinalize) {
  // Create command-buffer
  pi_ext_command_buffer CommandBuffer = nullptr;
  pi_ext_command_buffer_desc CommandBufferDesc;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferCreate>(
                Context, Device, &CommandBufferDesc, &CommandBuffer)),
            PI_SUCCESS);
  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferFinalize>(
          CommandBuffer)),
      PI_SUCCESS);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferRelease>(
                CommandBuffer)),
            PI_SUCCESS);
}

TEST_F(LevelZeroCommandBuffersTest, piextCommandBufferEnqueue) {
  // Create some pi_mem for passing to the kernel
  size_t MemSize = 1u;
  pi_mem MemObj;
  std::vector<int> InitData(MemSize, 0);
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                Context, PI_MEM_FLAGS_ACCESS_RW, MemSize * sizeof(int), nullptr,
                &MemObj, nullptr)),
            PI_SUCCESS);

  // Initialize the buffer on device
  pi_event WriteEvent;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
                Queue, MemObj, PI_FALSE, 0, MemSize * sizeof(int),
                InitData.data(), 0, nullptr, &WriteEvent)),
            PI_SUCCESS);
  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piEventsWait>(1, &WriteEvent)),
      PI_SUCCESS);

  // Create command-buffer
  pi_ext_command_buffer CommandBuffer = nullptr;
  pi_ext_command_buffer_desc CommandBufferDesc;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferCreate>(
                Context, Device, &CommandBufferDesc, &CommandBuffer)),
            PI_SUCCESS);

  // Build the kernel from the SPIR-V source in spirv-source.hpp
  pi_program Prog;

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramCreate>(
                Context, SpvSource, SpvSourceLen, &Prog)),
            PI_SUCCESS);

  Plugin->call_nocheck<detail::PiApiKind::piProgramBuild>(
      Prog, 1, &Device, nullptr, nullptr, nullptr);

  size_t KernelNameSize;

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramGetInfo>(
                Prog, pi_program_info::PI_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr,
                &KernelNameSize)),
            PI_SUCCESS);
  std::vector<char> KernelNames(KernelNameSize);
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramGetInfo>(
                Prog, pi_program_info::PI_PROGRAM_INFO_KERNEL_NAMES,
                KernelNameSize, KernelNames.data(), nullptr)),
            PI_SUCCESS);
  std::string KernelNameStr(KernelNames.data());

  pi_kernel Kernel;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piKernelCreate>(
                Prog, KernelNameStr.c_str(), &Kernel)),
            PI_SUCCESS);
  ASSERT_NE(Kernel, nullptr);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextKernelSetArgMemObj>(
                Kernel, 0, &MemObj)),
            PI_SUCCESS);

  pi_ext_sync_point SyncPoint;

  std::array<size_t, 3> GlobalWorkOffset = {0, 0, 0};
  std::array<size_t, 3> GlobalWorkSize = {1, 1, 1};
  std::array<size_t, 3> LocalWorkSize = {1, 1, 1};

  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferNDRangeKernel>(
          CommandBuffer, Kernel, 1, &GlobalWorkOffset[0], &GlobalWorkSize[0],
          &LocalWorkSize[0], 0, nullptr, &SyncPoint)),
      PI_SUCCESS);

  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferFinalize>(
          CommandBuffer)),
      PI_SUCCESS);

  pi_event CommandBufferEvent;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextEnqueueCommandBuffer>(
                CommandBuffer, Queue, 0, nullptr, &CommandBufferEvent)),
            PI_SUCCESS);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piEventsWait>(
                1, &CommandBufferEvent)),
            PI_SUCCESS);

  std::vector<int> HostData(MemSize);
  pi_event HostEvent;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                Queue, MemObj, PI_FALSE, 0, MemSize * sizeof(int),
                HostData.data(), 0, nullptr, &HostEvent)),
            PI_SUCCESS);

  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piEventsWait>(1, &HostEvent)),
      PI_SUCCESS);

  ASSERT_EQ(HostData[0], 1);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piMemRelease>(MemObj)),
            PI_SUCCESS);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferRelease>(
                CommandBuffer)),
            PI_SUCCESS);
}

TEST_F(LevelZeroCommandBuffersTest, piextCommandBufferEnqueueMultiple) {
  // Create some pi_mem for passing to the kernel
  size_t MemSize = 1u;
  pi_mem MemObj;
  std::vector<int> InitData(MemSize, 0);
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                Context, PI_MEM_FLAGS_ACCESS_RW, MemSize * sizeof(int), nullptr,
                &MemObj, nullptr)),
            PI_SUCCESS);

  // Initialize the buffer on device
  pi_event WriteEvent;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
                Queue, MemObj, PI_FALSE, 0, MemSize * sizeof(int),
                InitData.data(), 0, nullptr, &WriteEvent)),
            PI_SUCCESS);
  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piEventsWait>(1, &WriteEvent)),
      PI_SUCCESS);

  // Create command-buffer
  pi_ext_command_buffer CommandBuffer = nullptr;
  pi_ext_command_buffer_desc CommandBufferDesc;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferCreate>(
                Context, Device, &CommandBufferDesc, &CommandBuffer)),
            PI_SUCCESS);

  // Build the kernel from the SPIR-V source in spirv-source.hpp
  pi_program Prog;

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramCreate>(
                Context, SpvSource, SpvSourceLen, &Prog)),
            PI_SUCCESS);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramBuild>(
                Prog, 1, &Device, nullptr, nullptr, nullptr)),
            PI_SUCCESS);

  size_t KernelNameSize;

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramGetInfo>(
                Prog, pi_program_info::PI_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr,
                &KernelNameSize)),
            PI_SUCCESS);
  std::vector<char> KernelNames(KernelNameSize);
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piProgramGetInfo>(
                Prog, pi_program_info::PI_PROGRAM_INFO_KERNEL_NAMES,
                KernelNameSize, KernelNames.data(), nullptr)),
            PI_SUCCESS);
  std::string KernelNameStr(KernelNames.data());

  pi_kernel Kernel;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piKernelCreate>(
                Prog, KernelNameStr.c_str(), &Kernel)),
            PI_SUCCESS);
  ASSERT_NE(Kernel, nullptr);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextKernelSetArgMemObj>(
                Kernel, 0, &MemObj)),
            PI_SUCCESS);

  pi_ext_sync_point SyncPoint;

  std::array<size_t, 3> GlobalWorkOffset = {0, 0, 0};
  std::array<size_t, 3> GlobalWorkSize = {1, 1, 1};
  std::array<size_t, 3> LocalWorkSize = {1, 1, 1};

  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferNDRangeKernel>(
          CommandBuffer, Kernel, 1, &GlobalWorkOffset[0], &GlobalWorkSize[0],
          &LocalWorkSize[0], 0, nullptr, &SyncPoint)),
      PI_SUCCESS);

  // Enqueue the same kernel again to test sync points
  pi_ext_sync_point SyncPoint2;

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextKernelSetArgMemObj>(
                Kernel, 0, &MemObj)),
            PI_SUCCESS);
  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferNDRangeKernel>(
          CommandBuffer, Kernel, 1, &GlobalWorkOffset[0], &GlobalWorkSize[0],
          &LocalWorkSize[0], 1, &SyncPoint, &SyncPoint2)),
      PI_SUCCESS);

  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferFinalize>(
          CommandBuffer)),
      PI_SUCCESS);

  pi_event CommandBufferEvent;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextEnqueueCommandBuffer>(
                CommandBuffer, Queue, 0, nullptr, &CommandBufferEvent)),
            PI_SUCCESS);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piEventsWait>(
                1, &CommandBufferEvent)),
            PI_SUCCESS);

  std::vector<int> HostData(MemSize);
  pi_event HostEvent;
  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                Queue, MemObj, PI_FALSE, 0, MemSize * sizeof(int),
                HostData.data(), 0, nullptr, &HostEvent)),
            PI_SUCCESS);

  ASSERT_EQ(
      (Plugin->call_nocheck<detail::PiApiKind::piEventsWait>(1, &HostEvent)),
      PI_SUCCESS);

  ASSERT_EQ(HostData[0], 2);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piMemRelease>(MemObj)),
            PI_SUCCESS);

  ASSERT_EQ((Plugin->call_nocheck<detail::PiApiKind::piextCommandBufferRelease>(
                CommandBuffer)),
            PI_SUCCESS);
}
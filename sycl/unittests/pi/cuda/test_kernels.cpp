//==---- test_kernels.cpp --- PI unit tests --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlugin.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_cuda.hpp>

// PI CUDA kernels carry an additional argument for the implicit global offset.
#define NUM_IMPLICIT_ARGS 1

using namespace cl::sycl;

struct CudaKernelsTest : public ::testing::Test {

protected:
  detail::plugin plugin = pi::initializeAndGet(backend::cuda);
  pi_platform platform_;
  pi_device device_;
  pi_context context_;
  pi_queue queue_;

  void SetUp() override {
    pi_uint32 numPlatforms = 0;
    ASSERT_EQ(plugin.getBackend(), backend::cuda);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &numPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  numPlatforms, &platform_, nullptr)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piDevicesGet>(
                  platform_, PI_DEVICE_TYPE_GPU, 1, &device_, nullptr)),
              PI_SUCCESS);
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piContextCreate>(
                  nullptr, 1, &device_, nullptr, nullptr, &context_)),
              PI_SUCCESS);
    ASSERT_NE(context_, nullptr);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueCreate>(
                  context_, device_, 0, &queue_)),
              PI_SUCCESS);
    ASSERT_NE(queue_, nullptr);
    ASSERT_EQ(queue_->get_context(), context_);
  }

  void TearDown() override {
    plugin.call<detail::PiApiKind::piDeviceRelease>(device_);
    plugin.call<detail::PiApiKind::piQueueRelease>(queue_);
    plugin.call<detail::PiApiKind::piContextRelease>(context_);
  }

  CudaKernelsTest() = default;

  ~CudaKernelsTest() = default;
};

const char *ptxSource = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry _Z8myKernelPi(\n\
	.param .u64 _Z8myKernelPi_param_0\n\
)\n\
{\n\
	.reg .s32 	%r<5>;\n\
	.reg .s64 	%rd<5>;\n\
	ld.param.u64 	%rd1, [_Z8myKernelPi_param_0];\n\
	cvta.to.global.u64 	%rd2, %rd1;\n\
	.loc 1 3 1\n\
	mov.u32 	%r1, %ntid.x;\n\
	mov.u32 	%r2, %ctaid.x;\n\
	mov.u32 	%r3, %tid.x;\n\
	mad.lo.s32 	%r4, %r1, %r2, %r3;\n\
	mul.wide.s32 	%rd3, %r4, 4;\n\
	add.s64 	%rd4, %rd2, %rd3;\n\
	.loc 1 4 1\n\
	st.global.u32 	[%rd4], %r4;\n\
	.loc 1 5 2\n\
	ret;\n\
    ret;\
\n\
}\
\n\
";

const char *twoParams = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry twoParamKernel(\n\
	.param .u64 twoParamKernel_param_0,\n\
  .param .u64 twoParamKernel_param_1\n\
)\n\
{\n\
  ret;\
  \n\
}\n\
";

const char *threeParamsTwoLocal = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry twoParamKernelLocal(\n\
	.param .u64 twoParamKernel_param_0,\n\
  .param .u32 twoParamKernel_param_1,\n\
  .param .u32 twoParamKernel_param_2\n\
)\n\
{\n\
  ret;\
  \n\
}\n\
";

TEST_F(CudaKernelsTest, PICreateProgramAndKernel) {

  pi_program prog;
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
                context_, 1, &device_, nullptr,
                (const unsigned char **)&ptxSource, &binary_status, &prog)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "_Z8myKernelPi", &kern)),
            PI_SUCCESS);
  ASSERT_NE(kern, nullptr);
}

TEST_F(CudaKernelsTest, PIKernelArgumentSimple) {

  pi_program prog;
  /// NOTE: `binary_status` currently unsused in the CUDA backend but in case we
  /// use it at some point in the future, pass it anyway and check the result.
  /// Same goes for all the other tests in this file.
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
                context_, 1, &device_, nullptr,
                (const unsigned char **)&ptxSource, &binary_status, &prog)),
            PI_SUCCESS);
  ASSERT_EQ(binary_status, PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "_Z8myKernelPi", &kern)),
            PI_SUCCESS);

  int number = 10;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 0, sizeof(int), &number)),
            PI_SUCCESS);
  const auto &kernArgs = kern->get_arg_indices();
  ASSERT_EQ(kernArgs.size(), (size_t)1 + NUM_IMPLICIT_ARGS);
  int storedValue = *(static_cast<const int *>(kernArgs[0]));
  ASSERT_EQ(storedValue, number);
}

TEST_F(CudaKernelsTest, PIKernelArgumentSetTwice) {

  pi_program prog;
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
                context_, 1, &device_, nullptr,
                (const unsigned char **)&ptxSource, &binary_status, &prog)),
            PI_SUCCESS);
  ASSERT_EQ(binary_status, PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "_Z8myKernelPi", &kern)),
            PI_SUCCESS);

  int number = 10;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 0, sizeof(int), &number)),
            PI_SUCCESS);
  const auto &kernArgs = kern->get_arg_indices();
  ASSERT_GT(kernArgs.size(), (size_t)0 + NUM_IMPLICIT_ARGS);
  int storedValue = *(static_cast<const int *>(kernArgs[0]));
  ASSERT_EQ(storedValue, number);

  int otherNumber = 934;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 0, sizeof(int), &otherNumber)),
            PI_SUCCESS);
  const auto &kernArgs2 = kern->get_arg_indices();
  ASSERT_EQ(kernArgs2.size(), (size_t)1 + NUM_IMPLICIT_ARGS);
  storedValue = *(static_cast<const int *>(kernArgs2[0]));
  ASSERT_EQ(storedValue, otherNumber);
}

TEST_F(CudaKernelsTest, PIKernelSetMemObj) {

  pi_program prog;
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
                context_, 1, &device_, nullptr,
                (const unsigned char **)&ptxSource, &binary_status, &prog)),
            PI_SUCCESS);
  ASSERT_EQ(binary_status, PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "_Z8myKernelPi", &kern)),
            PI_SUCCESS);

  size_t memSize = 1024u;
  pi_mem memObj;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj,
                nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 0, sizeof(pi_mem), &memObj)),
            PI_SUCCESS);
  const auto &kernArgs = kern->get_arg_indices();
  ASSERT_EQ(kernArgs.size(), (size_t)1 + NUM_IMPLICIT_ARGS);
  pi_mem storedValue = *(static_cast<pi_mem *>(kernArgs[0]));
  ASSERT_EQ(storedValue, memObj);
}

TEST_F(CudaKernelsTest, PIkerneldispatch) {

  pi_program prog;
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
                context_, 1, &device_, nullptr,
                (const unsigned char **)&ptxSource, &binary_status, &prog)),
            PI_SUCCESS);
  ASSERT_EQ(binary_status, PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "_Z8myKernelPi", &kern)),
            PI_SUCCESS);

  size_t memSize = 1024u;
  pi_mem memObj;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj,
                nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piextKernelSetArgMemObj>(
                kern, 0, &memObj)),
            PI_SUCCESS);

  size_t workDim = 1;
  size_t globalWorkOffset[] = {0};
  size_t globalWorkSize[] = {1};
  size_t localWorkSize[] = {1};
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueKernelLaunch>(
                queue_, kern, workDim, globalWorkOffset, globalWorkSize,
                localWorkSize, 0, nullptr, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
}

TEST_F(CudaKernelsTest, PIkerneldispatchTwo) {

  pi_program prog;
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
                context_, 1, &device_, nullptr,
                (const unsigned char **)&twoParams, &binary_status, &prog)),
            PI_SUCCESS);
  ASSERT_EQ(binary_status, PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "twoParamKernel", &kern)),
            PI_SUCCESS);

  size_t memSize = 1024u;
  pi_mem memObj;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj,
                nullptr)),
            PI_SUCCESS);

  pi_mem memObj2;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj2,
                nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piextKernelSetArgMemObj>(
                kern, 0, &memObj)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piextKernelSetArgMemObj>(
                kern, 1, &memObj2)),
            PI_SUCCESS);

  size_t workDim = 1;
  size_t globalWorkOffset[] = {0};
  size_t globalWorkSize[] = {1};
  size_t localWorkSize[] = {1};
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueKernelLaunch>(
                queue_, kern, workDim, globalWorkOffset, globalWorkSize,
                localWorkSize, 0, nullptr, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemRelease>(memObj)),
            PI_SUCCESS);
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemRelease>(memObj2)),
            PI_SUCCESS);
}

TEST_F(CudaKernelsTest, PIKernelArgumentSetTwiceOneLocal) {

  pi_program prog;
  pi_int32 binary_status = PI_SUCCESS;
  ASSERT_EQ(
      (plugin.call_nocheck<detail::PiApiKind::piProgramCreateWithBinary>(
          context_, 1, &device_, nullptr,
          (const unsigned char **)&threeParamsTwoLocal, &binary_status, &prog)),
      PI_SUCCESS);
  ASSERT_EQ(binary_status, PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piProgramBuild>(
                prog, 1, &device_, "", nullptr, nullptr)),
            PI_SUCCESS);

  pi_kernel kern;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelCreate>(
                prog, "twoParamKernelLocal", &kern)),
            PI_SUCCESS);

  int number = 10;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 0, sizeof(int), &number)),
            PI_SUCCESS);
  const auto &kernArgs = kern->get_arg_indices();
  ASSERT_GT(kernArgs.size(), (size_t)0 + NUM_IMPLICIT_ARGS);
  int storedValue = *(static_cast<const int *>(kernArgs[0]));
  ASSERT_EQ(storedValue, number);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 1, sizeof(int), nullptr)),
            PI_SUCCESS);
  const auto &kernArgs2 = kern->get_arg_indices();
  ASSERT_EQ(kernArgs2.size(), (size_t)2 + NUM_IMPLICIT_ARGS);
  storedValue = *(static_cast<const int *>(kernArgs2[1]));
  ASSERT_EQ(storedValue, 0);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piKernelSetArg>(
                kern, 2, sizeof(int), nullptr)),
            PI_SUCCESS);
  const auto &kernArgs3 = kern->get_arg_indices();
  ASSERT_EQ(kernArgs3.size(), (size_t)3 + NUM_IMPLICIT_ARGS);
  storedValue = *(static_cast<const int *>(kernArgs3[2]));
  ASSERT_EQ(storedValue, static_cast<int>(sizeof(int)));
}

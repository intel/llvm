//==---- test_mem_obj.cpp --- PI unit tests --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <pi_cuda.hpp>

using namespace cl::sycl;

struct DISABLED_CudaTestMemObj : public ::testing::Test {

protected:
  pi_platform platform_;
  pi_device device_;
  pi_context context_;

  void SetUp() override {
    cuCtxSetCurrent(nullptr);
    pi_uint32 numPlatforms = 0;

    ASSERT_EQ((PI_CALL_NOCHECK(piPlatformsGet)(0, nullptr, &numPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ(
        (PI_CALL_NOCHECK(piPlatformsGet)(numPlatforms, &platform_, nullptr)),
        PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((PI_CALL_NOCHECK(piDevicesGet)(platform_, PI_DEVICE_TYPE_GPU, 1, &device_, nullptr)), PI_SUCCESS);
    ASSERT_EQ((PI_CALL_NOCHECK(piContextCreate)(nullptr, 1, &device_, nullptr, nullptr, &context_)), PI_SUCCESS);
    EXPECT_NE(context_, nullptr);
  }

  void TearDown() override {
    PI_CALL(piDeviceRelease)(device_);
    PI_CALL(piContextRelease)(context_);
  }

  DISABLED_CudaTestMemObj() { detail::pi::initialize(); }

  ~DISABLED_CudaTestMemObj() = default;
};

TEST_F(DISABLED_CudaTestMemObj, piMemBufferCreateSimple) {
  const size_t memSize = 1024u;
  pi_mem memObj;
  ASSERT_EQ((PI_CALL_NOCHECK(piMemBufferCreate)(context_, PI_MEM_FLAGS_ACCESS_RW, memSize, nullptr, &memObj)), PI_SUCCESS);

  ASSERT_EQ((PI_CALL_NOCHECK(piMemRelease)(memObj)), PI_SUCCESS);
}

TEST_F(DISABLED_CudaTestMemObj, piMemBufferCreateNoActiveContext) {
  const size_t memSize = 1024u;
  // Context has been destroyed

  CUcontext current = nullptr;

  // pop CUDA contexts until there is not a cuda context bound to the thread
  do {
    CUcontext oldContext = nullptr;
    auto cuErr = cuCtxPopCurrent(&oldContext);
    EXPECT_EQ(cuErr, CUDA_SUCCESS);

    // There should not be any active CUDA context
    cuErr = cuCtxGetCurrent(&current);
    ASSERT_EQ(cuErr, CUDA_SUCCESS);
  } while (current != nullptr);

  // The context object is passed, even if its not active it should be used
  // to allocate the memory object
  pi_mem memObj;
  ASSERT_EQ((PI_CALL_NOCHECK(piMemBufferCreate)(context_, PI_MEM_FLAGS_ACCESS_RW,
                                        memSize, nullptr, &memObj)), PI_SUCCESS);
  ASSERT_NE(memObj, nullptr);

  ASSERT_EQ((PI_CALL_NOCHECK(piMemRelease)(memObj)), PI_SUCCESS);
}

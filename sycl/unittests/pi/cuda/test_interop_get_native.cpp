//==------- test_interop_get_native.cpp - SYCL CUDA get_native tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestGetPlatforms.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <cuda.h>
#include <iostream>

using namespace cl::sycl;

struct CudaInteropGetNativeTests : public ::testing::TestWithParam<platform> {

protected:
  queue syclQueue_;
  context syclContext_;
  device syclDevice_;

  void SetUp() override {
    syclDevice_ = GetParam().get_devices()[0];
    syclQueue_ = queue{syclDevice_};
    syclContext_ = syclQueue_.get_context();
  }

  void TearDown() override {}
};

TEST_P(CudaInteropGetNativeTests, getNativeDevice) {
  CUdevice cudaDevice = get_native<backend::cuda>(syclDevice_);
  char cudaDeviceName[2] = {0, 0};
  CUresult result = cuDeviceGetName(cudaDeviceName, 2, cudaDevice);
  ASSERT_EQ(result, CUDA_SUCCESS);
  ASSERT_NE(cudaDeviceName[0], 0);
}

TEST_P(CudaInteropGetNativeTests, getNativeContext) {
  CUcontext cudaContext = get_native<backend::cuda>(syclContext_);
  ASSERT_NE(cudaContext, nullptr);
}

TEST_P(CudaInteropGetNativeTests, getNativeQueue) {
  CUstream cudaStream = get_native<backend::cuda>(syclQueue_);
  ASSERT_NE(cudaStream, nullptr);

  CUcontext streamContext = nullptr;
  CUresult result = cuStreamGetCtx(cudaStream, &streamContext);
  ASSERT_EQ(result, CUDA_SUCCESS);

  CUcontext cudaContext = get_native<backend::cuda>(syclContext_);
  ASSERT_EQ(streamContext, cudaContext);
}

TEST_P(CudaInteropGetNativeTests, interopTaskGetMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_.submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.interop_task([=](interop_handler ih) {
      CUdeviceptr cudaPtr = ih.get_mem<backend::cuda>(syclAccessor);
      CUdeviceptr cudaPtrBase;
      size_t cudaPtrSize = 0;
      CUcontext cudaContext = get_native<backend::cuda>(syclContext_);
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(cudaContext));
      ASSERT_EQ(CUDA_SUCCESS,
                cuMemGetAddressRange(&cudaPtrBase, &cudaPtrSize, cudaPtr));
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), cudaPtrSize);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, interopTaskGetBufferMem) {
  CUstream cudaStream = get_native<backend::cuda>(syclQueue_);
  syclQueue_.submit([&](handler &cgh) {
    cgh.interop_task([=](interop_handler ih) {
      CUstream cudaInteropStream = ih.get_queue<backend::cuda>();
      ASSERT_EQ(cudaInteropStream, cudaStream);
    });
  });
}

INSTANTIATE_TEST_CASE_P(
    OnCudaPlatform, CudaInteropGetNativeTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("CUDA BACKEND")), );

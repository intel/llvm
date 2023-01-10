//==------- test_interop_get_native.cpp - SYCL CUDA get_native tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlatforms.hpp"
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

struct CudaInteropGetNativeTests : public ::testing::TestWithParam<platform> {

protected:
  std::unique_ptr<queue> syclQueue_;
  device syclDevice_;

  void SetUp() override {
    syclDevice_ = GetParam().get_devices()[0];
    syclQueue_ = std::unique_ptr<queue>{new queue{syclDevice_}};
  }

  void TearDown() override { syclQueue_.reset(); }
};

TEST_P(CudaInteropGetNativeTests, getNativeDevice) {
  CUdevice cudaDevice = get_native<backend::ext_oneapi_cuda>(syclDevice_);
  char cudaDeviceName[2] = {0, 0};
  CUresult result = cuDeviceGetName(cudaDeviceName, 2, cudaDevice);
  ASSERT_EQ(result, CUDA_SUCCESS);
  ASSERT_NE(cudaDeviceName[0], 0);
}

TEST_P(CudaInteropGetNativeTests, getNativeContext) {
  CUcontext cudaContext =
      get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
  ASSERT_NE(cudaContext, nullptr);
}

TEST_P(CudaInteropGetNativeTests, getNativeQueue) {
  CUstream cudaStream = get_native<backend::ext_oneapi_cuda>(*syclQueue_);
  ASSERT_NE(cudaStream, nullptr);

  CUcontext streamContext = nullptr;
  CUresult result = cuStreamGetCtx(cudaStream, &streamContext);
  ASSERT_EQ(result, CUDA_SUCCESS);

  CUcontext cudaContext =
      get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
  ASSERT_EQ(streamContext, cudaContext);
}

TEST_P(CudaInteropGetNativeTests, interopTaskGetMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_->submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.host_task([=](interop_handle ih) {
      CUdeviceptr cudaPtr =
          ih.get_native_mem<backend::ext_oneapi_cuda>(syclAccessor);
      CUdeviceptr cudaPtrBase;
      size_t cudaPtrSize = 0;
      CUcontext cudaContext =
          get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(cudaContext));
      ASSERT_EQ(CUDA_SUCCESS,
                cuMemGetAddressRange(&cudaPtrBase, &cudaPtrSize, cudaPtr));
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), cudaPtrSize);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, interopTaskGetQueue) {
  CUstream cudaStream = get_native<backend::ext_oneapi_cuda>(*syclQueue_);
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      CUstream cudaInteropStream =
          ih.get_native_queue<backend::ext_oneapi_cuda>();
      ASSERT_EQ(cudaInteropStream, cudaStream);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, hostTaskGetNativeMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_->submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.host_task([=](interop_handle ih) {
      CUdeviceptr cudaPtr =
          ih.get_native_mem<backend::ext_oneapi_cuda>(syclAccessor);
      CUdeviceptr cudaPtrBase;
      size_t cudaPtrSize = 0;
      CUcontext cudaContext =
          get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(cudaContext));
      ASSERT_EQ(CUDA_SUCCESS,
                cuMemGetAddressRange(&cudaPtrBase, &cudaPtrSize, cudaPtr));
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), cudaPtrSize);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, hostTaskGetNativeQueue) {
  CUstream cudaStream = get_native<backend::ext_oneapi_cuda>(*syclQueue_);
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      CUstream cudaInteropStream =
          ih.get_native_queue<backend::ext_oneapi_cuda>();
      ASSERT_EQ(cudaInteropStream, cudaStream);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, hostTaskGetNativeContext) {
  CUcontext cudaContext =
      get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      CUcontext cudaInteropContext =
          ih.get_native_context<backend::ext_oneapi_cuda>();
      ASSERT_EQ(cudaInteropContext, cudaContext);
    });
  });
}

INSTANTIATE_TEST_SUITE_P(
    OnCudaPlatform, CudaInteropGetNativeTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("CUDA BACKEND")));

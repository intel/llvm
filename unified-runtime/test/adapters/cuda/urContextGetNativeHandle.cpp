// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urCudaContextGetNativeHandle = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCudaContextGetNativeHandle);

TEST_P(urCudaContextGetNativeHandle, Success) {
  ur_native_handle_t native_context = 0;
  ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_context));
  CUcontext cuda_context = reinterpret_cast<CUcontext>(native_context);

  unsigned int cudaVersion;
  ASSERT_SUCCESS_CUDA(cuCtxGetApiVersion(cuda_context, &cudaVersion));
}

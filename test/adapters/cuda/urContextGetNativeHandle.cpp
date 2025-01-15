// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urCudaContextGetNativeHandle = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaContextGetNativeHandle);

TEST_P(urCudaContextGetNativeHandle, Success) {
  ur_native_handle_t native_context = 0;
  ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_context));
  CUcontext cuda_context = reinterpret_cast<CUcontext>(native_context);

  unsigned int cudaVersion;
  ASSERT_SUCCESS_CUDA(cuCtxGetApiVersion(cuda_context, &cudaVersion));
}

// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urKernelSetArgMemObjTest : uur::urKernelTest {
  void SetUp() {
    program_name = "fill";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     16 * sizeof(uint32_t), nullptr, &buffer));
  }

  void TearDown() {
    if (buffer) {
      ASSERT_SUCCESS(urMemRelease(buffer));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::TearDown());
  }

  ur_mem_handle_t buffer = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetArgMemObjTest);

TEST_P(urKernelSetArgMemObjTest, Success) {
  ASSERT_SUCCESS(urKernelSetArgMemObj(kernel, 0, nullptr, buffer));
}

TEST_P(urKernelSetArgMemObjTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelSetArgMemObj(nullptr, 0, nullptr, buffer));
}

TEST_P(urKernelSetArgMemObjTest, InvalidKernelArgumentIndex) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  uint32_t num_kernel_args = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                 sizeof(num_kernel_args), &num_kernel_args,
                                 nullptr));
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
      urKernelSetArgMemObj(kernel, num_kernel_args + 1, nullptr, buffer));
}

TEST_P(urKernelSetArgMemObjTest, InvalidEnumeration) {
  ur_kernel_arg_mem_obj_properties_t props{
      UR_STRUCTURE_TYPE_KERNEL_ARG_MEM_OBJ_PROPERTIES, /* stype */
      nullptr,                                         /* pNext */
      UR_MEM_FLAG_FORCE_UINT32                         /* memoryAccess */
  };
  ASSERT_EQ_RESULT(urKernelSetArgMemObj(kernel, 0, &props, buffer),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
}

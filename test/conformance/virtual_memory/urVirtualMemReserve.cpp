// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemReserveTestWithParam =
    uur::urVirtualMemGranularityTestWithParam<size_t>;
UUR_DEVICE_TEST_SUITE_P(urVirtualMemReserveTestWithParam,
                        ::testing::Values(2, 4, 8, 16, 32, 64, 128, 256, 512,
                                          1024, 2048, 5000, 100000),
                        uur::deviceTestWithParamPrinter<size_t>);

TEST_P(urVirtualMemReserveTestWithParam, SuccessNoStartPointer) {
  // round up to nearest granularity
  size_t virtual_mem_size =
      uur::RoundUpToNearestFactor(getParam(), granularity);
  void *virtual_mem_start = nullptr;
  ASSERT_SUCCESS(urVirtualMemReserve(context, nullptr, virtual_mem_size,
                                     &virtual_mem_start));
  ASSERT_NE(virtual_mem_start, nullptr);

  EXPECT_SUCCESS(
      urVirtualMemFree(context, virtual_mem_start, virtual_mem_size));
}

TEST_P(urVirtualMemReserveTestWithParam, SuccessWithStartPointer) {
  // roundup to nearest granularity
  size_t page_size = uur::RoundUpToNearestFactor(getParam(), granularity);
  void *origin_ptr = nullptr;
  ASSERT_SUCCESS(urVirtualMemReserve(context, nullptr, page_size, &origin_ptr));
  ASSERT_NE(origin_ptr, nullptr);

  // try to reserve at the end of origin_ptr
  void *virtual_mem_ptr = nullptr;
  void *pStart = (uint8_t *)origin_ptr + page_size;
  ASSERT_SUCCESS(
      urVirtualMemReserve(context, pStart, page_size, &virtual_mem_ptr));
  ASSERT_NE(virtual_mem_ptr, nullptr);

  // both pointers have to be freed
  EXPECT_SUCCESS(urVirtualMemFree(context, origin_ptr, page_size));
  EXPECT_SUCCESS(urVirtualMemFree(context, virtual_mem_ptr, page_size));
}

using urVirtualMemReserveTest = uur::urVirtualMemGranularityTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemReserveTest);

TEST_P(urVirtualMemReserveTest, InvalidNullHandleContext) {
  size_t page_size = uur::RoundUpToNearestFactor(1024, granularity);
  void *virtual_ptr = nullptr;
  ASSERT_EQ_RESULT(
      urVirtualMemReserve(nullptr, nullptr, page_size, &virtual_ptr),
      UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemReserveTest, InvalidNullPointer) {
  size_t page_size = uur::RoundUpToNearestFactor(1024, granularity);
  ASSERT_EQ_RESULT(urVirtualMemReserve(context, nullptr, page_size, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

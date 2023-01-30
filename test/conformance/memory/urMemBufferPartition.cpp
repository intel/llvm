// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urMemBufferPartitionTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferPartitionTest);

TEST_P(urMemBufferPartitionTest, Success) {
  ur_buffer_region_t region{0, 1024};
  ur_mem_handle_t partition = nullptr;
  ASSERT_SUCCESS(urMemBufferPartition(buffer, UR_MEM_FLAG_READ_WRITE,
                                      UR_BUFFER_CREATE_TYPE_REGION, &region,
                                      &partition));
  ASSERT_NE(partition, nullptr);
}

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urMemImageCreateWithNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemImageCreateWithNativeHandleTest);

TEST_P(urMemImageCreateWithNativeHandleTest, InvalidNullHandleNativeMem) {
    ur_mem_handle_t mem = nullptr;
    ur_image_format_t imageFormat = {
        /*.channelOrder =*/UR_IMAGE_CHANNEL_ORDER_ARGB,
        /*.channelType =*/UR_IMAGE_CHANNEL_TYPE_UNORM_INT8,
    };
    ur_image_desc_t imageDesc = {
        /*.stype =*/UR_STRUCTURE_TYPE_IMAGE_DESC,
        /*.pNext =*/nullptr,
        /*.type =*/UR_MEM_TYPE_IMAGE2D,
        /*.width =*/16,
        /*.height =*/16,
        /*.depth =*/1,
        /*.arraySize =*/1,
        /*.rowPitch =*/16,
        /*.slicePitch =*/16 * 16,
        /*.numMipLevel =*/0,
        /*.numSamples =*/0,
    };
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemImageCreateWithNativeHandle(nullptr, context,
                                                      &imageFormat, &imageDesc,
                                                      nullptr, &mem));
}

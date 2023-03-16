// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urUSMPoolCreateTest = uur::urContextTest;

TEST_F(urUSMPoolCreateTest, Success) {
    ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                 UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
    ur_usm_pool_handle_t pool = nullptr;
    ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
    ASSERT_NE(pool, nullptr);
    EXPECT_SUCCESS(urUSMPoolDestroy(context, pool));
}

TEST_F(urUSMPoolCreateTest, InvalidNullHandleContext) {
    ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                 UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
    ur_usm_pool_handle_t pool = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMPoolCreate(nullptr, &pool_desc, &pool));
}

TEST_F(urUSMPoolCreateTest, InvalidNullPointerPoolDesc) {
    ur_usm_pool_handle_t pool = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urUSMPoolCreate(context, nullptr, &pool));
}

TEST_F(urUSMPoolCreateTest, InvalidNullPointerPool) {
    ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                 UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urUSMPoolCreate(context, &pool_desc, nullptr));
}

TEST_F(urUSMPoolCreateTest, InvalidEnumerationFlags) {
    ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                 UR_USM_POOL_FLAG_FORCE_UINT32};
    ur_usm_pool_handle_t pool = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urUSMPoolCreate(context, &pool_desc, &pool));
}

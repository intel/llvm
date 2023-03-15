// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urUSMPoolDestroyTest : uur::urQueueTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());
        ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                     UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
        ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
        ASSERT_NE(pool, nullptr);
    }

    void TearDown() override {
        if (pool) {
            ASSERT_SUCCESS(urUSMPoolDestroy(context, pool));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
    }

    ur_usm_pool_handle_t pool = nullptr;
};

TEST_F(urUSMPoolDestroyTest, Success) {
    ASSERT_SUCCESS(urUSMPoolDestroy(context, pool));
    pool = nullptr; // prevent double-delete
}

TEST_F(urUSMPoolDestroyTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMPoolDestroy(nullptr, pool));
}

TEST_F(urUSMPoolDestroyTest, InvalidNullHandlePool) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMPoolDestroy(context, nullptr));
}

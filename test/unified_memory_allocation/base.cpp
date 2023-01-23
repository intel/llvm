// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "helpers.h"

#include <uma/base.h>

TEST_F(umaTest, versionEncodeDecode) {
    auto encoded = UMA_MAKE_VERSION(0, 9);
    ASSERT_EQ(UMA_MAJOR_VERSION(encoded), 0);
    ASSERT_EQ(UMA_MINOR_VERSION(encoded), 9);
}

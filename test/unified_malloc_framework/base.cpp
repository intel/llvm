// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "base.hpp"

#include <umf/base.h>

using umf_test::test;

TEST_F(test, versionEncodeDecode) {
    auto encoded = UMF_MAKE_VERSION(0, 9);
    ASSERT_EQ(UMF_MAJOR_VERSION(encoded), 0);
    ASSERT_EQ(UMF_MINOR_VERSION(encoded), 9);
}

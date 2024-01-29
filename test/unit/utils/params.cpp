// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "print.h"
#include "ur_print.hpp"

TYPED_TEST(ParamsTest, Print) {
    std::ostringstream out;
    out << this->params.get_struct();
    EXPECT_THAT(out.str(), MatchesRegex(this->params.get_expected()));
}

TEST(PrintPtr, nested_void_ptrs) {
    void *real = (void *)0xFEEDCAFEull;
    void **preal = &real;
    void ***ppreal = &preal;
    void ****pppreal = &ppreal;
    std::ostringstream out;
    ur::details::printPtr(out, pppreal);
    EXPECT_THAT(out.str(), MatchesRegex(".+ \\(.+ \\(.+ \\(.+\\)\\)\\)"));
}

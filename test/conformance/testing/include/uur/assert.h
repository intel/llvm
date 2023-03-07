// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UUR_CONFORMANCE_TESTING_ASSERT_H_INCLUDED
#define UUR_CONFORMANCE_TESTING_ASSERT_H_INCLUDED

#include <cstdio>
#include <cstdlib>

#define UUR_ASSERT(CONDITION, MESSAGE)                                     \
    if (!(CONDITION)) {                                                    \
        std::fprintf(stderr, "%s: %d: %s\n", __FILE__, __LINE__, MESSAGE); \
        std::abort();                                                      \
    }

#define UUR_ABORT(FORMAT, ...)                                        \
    {                                                                 \
        std::fprintf(stderr, "abort: %s: %d: " FORMAT "\n", __FILE__, \
                     __LINE__, __VA_ARGS__);                          \
        std::abort();                                                 \
    }

#endif // UUR_CONFORMANCE_TESTING_ASSERT_H_INCLUDED

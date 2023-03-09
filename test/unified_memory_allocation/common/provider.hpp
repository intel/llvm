/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UMA_TEST_PROVIDER_HPP
#define UMA_TEST_PROVIDER_HPP 1

#include <uma/base.h>
#include <uma/memory_provider.h>

#include <gtest/gtest.h>

#include "base.hpp"
#include "uma_helpers.hpp"

namespace uma_test {

auto wrapProviderUnique(uma_memory_provider_handle_t hProvider) {
    return uma::provider_unique_handle_t(hProvider, &umaMemoryProviderDestroy);
}

struct provider_base {
    uma_result_t initialize() noexcept {
        return UMA_RESULT_SUCCESS;
    };
    enum uma_result_t alloc(size_t, size_t, void **) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
    enum uma_result_t free(void *ptr, size_t size) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
    enum uma_result_t get_last_result(const char **) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
};

} // namespace uma_test

#endif /* UMA_TEST_PROVIDER_HPP */

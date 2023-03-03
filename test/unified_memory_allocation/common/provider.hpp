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
    uma_result_t initialize() noexcept { return UMA_RESULT_SUCCESS; };
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

struct provider_malloc : public provider_base {
    enum uma_result_t alloc(size_t size, size_t align, void **ptr) noexcept {
        if (!align) {
            *ptr = malloc(size);
            return (*ptr) ? UMA_RESULT_SUCCESS
                          : UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

#ifdef _WIN32
        // we could use _aligned_alloc but it requires using _aligned_free...
        return UMA_RESULT_ERROR_INVALID_ARGUMENT;
#else
        *ptr = ::aligned_alloc(align, size);
        return (*ptr) ? UMA_RESULT_SUCCESS
                      : UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
#endif
    }
    enum uma_result_t free(void *ptr, size_t) noexcept {
        ::free(ptr);
        return UMA_RESULT_SUCCESS;
    }
};

} // namespace uma_test

#endif /* UMA_TEST_PROVIDER_HPP */

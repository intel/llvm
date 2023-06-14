/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    enum uma_result_t get_recommended_page_size(size_t size,
                                                size_t *pageSize) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
    enum uma_result_t get_min_page_size(void *ptr, size_t *pageSize) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
    enum uma_result_t purge_lazy(void *ptr, size_t size) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
    enum uma_result_t purge_force(void *ptr, size_t size) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
    void get_name(const char **ppName) noexcept { *ppName = "base"; }
};

struct provider_malloc : public provider_base {
    enum uma_result_t alloc(size_t size, size_t align, void **ptr) noexcept {
        if (!align) {
            align = 8;
        }

#ifdef _WIN32
        *ptr = _aligned_malloc(size, align);
#else
        *ptr = ::aligned_alloc(align, size);
#endif

        return (*ptr) ? UMA_RESULT_SUCCESS
                      : UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    enum uma_result_t free(void *ptr, size_t) noexcept {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
        return UMA_RESULT_SUCCESS;
    }
    void get_name(const char **ppName) noexcept { *ppName = "malloc"; }
};

} // namespace uma_test

#endif /* UMA_TEST_PROVIDER_HPP */

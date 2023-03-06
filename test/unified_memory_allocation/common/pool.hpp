/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UMA_TEST_POOL_HPP
#define UMA_TEST_POOL_HPP 1

#include <malloc.h>
#include <uma/base.h>
#include <uma/memory_provider.h>

#include <gtest/gtest.h>
#include <stdlib.h>

#include "base.hpp"
#include "uma_helpers.hpp"

namespace uma_test {

auto wrapPoolUnique(uma_memory_pool_handle_t hPool) {
    return uma::pool_unique_handle_t(hPool, &umaPoolDestroy);
}

struct pool_base {
    uma_result_t initialize() noexcept {
        return UMA_RESULT_SUCCESS;
    };
    void *malloc(size_t size) noexcept {
        return nullptr;
    }
    void *calloc(size_t, size_t) noexcept {
        return nullptr;
    }
    void *realloc(void *, size_t) noexcept {
        return nullptr;
    }
    void *aligned_malloc(size_t, size_t) noexcept {
        return nullptr;
    }
    size_t malloc_usable_size(void *) noexcept {
        return 0;
    }
    void free(void *) noexcept {
    }
    enum uma_result_t get_last_result(const char **ppMessage) noexcept {
        return UMA_RESULT_ERROR_UNKNOWN;
    }
};

struct malloc_pool : public pool_base {
    void *malloc(size_t size) noexcept {
        return ::malloc(size);
    }
    void *calloc(size_t num, size_t size) noexcept {
        return ::calloc(num, size);
    }
    void *realloc(void *ptr, size_t size) noexcept {
        return ::realloc(ptr, size);
    }
    void *aligned_malloc(size_t size, size_t alignment) noexcept {
#ifdef _WIN32
        // we could use _aligned_alloc but it requires using _aligned_free...
        return nullptr;
#else
        return ::aligned_alloc(alignment, size);
#endif
    }
    size_t malloc_usable_size(void *ptr) noexcept {
#ifdef _WIN32
        return _msize(ptr);
#else
        return ::malloc_usable_size(ptr);
#endif
    }
    void free(void *ptr) noexcept {
        return ::free(ptr);
    }
};

} // namespace uma_test

#endif /* UMA_TEST_POOL_HPP */

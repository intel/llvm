/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_TEST_PROVIDER_HPP
#define UMF_TEST_PROVIDER_HPP 1

#include <umf/base.h>
#include <umf/memory_provider.h>

#include <gtest/gtest.h>

#include "base.hpp"
#include "umf_helpers.hpp"

namespace umf_test {

auto wrapProviderUnique(umf_memory_provider_handle_t hProvider) {
    return umf::provider_unique_handle_t(hProvider, &umfMemoryProviderDestroy);
}

struct provider_base {
    umf_result_t initialize() noexcept { return UMF_RESULT_SUCCESS; };
    enum umf_result_t alloc(size_t, size_t, void **) noexcept {
        return UMF_RESULT_ERROR_UNKNOWN;
    }
    enum umf_result_t free([[maybe_unused]] void *ptr,
                           [[maybe_unused]] size_t size) noexcept {
        return UMF_RESULT_ERROR_UNKNOWN;
    }
    void get_last_native_error(const char **, int32_t *) noexcept {}
    enum umf_result_t
    get_recommended_page_size([[maybe_unused]] size_t size,
                              [[maybe_unused]] size_t *pageSize) noexcept {
        return UMF_RESULT_ERROR_UNKNOWN;
    }
    enum umf_result_t
    get_min_page_size([[maybe_unused]] void *ptr,
                      [[maybe_unused]] size_t *pageSize) noexcept {
        return UMF_RESULT_ERROR_UNKNOWN;
    }
    enum umf_result_t purge_lazy([[maybe_unused]] void *ptr,
                                 [[maybe_unused]] size_t size) noexcept {
        return UMF_RESULT_ERROR_UNKNOWN;
    }
    enum umf_result_t purge_force([[maybe_unused]] void *ptr,
                                  [[maybe_unused]] size_t size) noexcept {
        return UMF_RESULT_ERROR_UNKNOWN;
    }
    const char *get_name() noexcept { return "base"; }
};

struct provider_malloc : public provider_base {
    enum umf_result_t alloc(size_t size, size_t align, void **ptr) noexcept {
        if (!align) {
            align = 8;
        }

#ifdef _WIN32
        *ptr = _aligned_malloc(size, align);
#else
        *ptr = ::aligned_alloc(align, size);
#endif

        return (*ptr) ? UMF_RESULT_SUCCESS
                      : UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    enum umf_result_t free(void *ptr, size_t) noexcept {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
        return UMF_RESULT_SUCCESS;
    }
    const char *get_name() noexcept { return "malloc"; }
};

struct provider_mock_out_of_mem : public provider_base {
    provider_malloc helper_prov;
    int allocNum = 0;
    umf_result_t initialize(int allocNum) noexcept {
        this->allocNum = allocNum;
        return UMF_RESULT_SUCCESS;
    }
    enum umf_result_t alloc(size_t size, size_t align, void **ptr) noexcept {
        if (allocNum <= 0) {
            *ptr = nullptr;
            return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        allocNum--;

        return helper_prov.alloc(size, align, ptr);
    }
    enum umf_result_t free(void *ptr, size_t size) noexcept {
        return helper_prov.free(ptr, size);
    }
    const char *get_name() noexcept { return "mock_out_of_mem"; }
};

} // namespace umf_test

#endif /* UMF_TEST_PROVIDER_HPP */

/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMA_HELPERS_H
#define UMA_HELPERS_H 1

#include <uma/memory_pool.h>
#include <uma/memory_pool_ops.h>
#include <uma/memory_provider.h>
#include <uma/memory_provider_ops.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace uma {

using pool_unique_handle_t =
    std::unique_ptr<uma_memory_pool_t,
                    std::function<void(uma_memory_pool_handle_t)>>;
using provider_unique_handle_t =
    std::unique_ptr<uma_memory_provider_t,
                    std::function<void(uma_memory_provider_handle_t)>>;

#define UMA_ASSIGN_OP(ops, type, func, default_return)                         \
    ops.func = [](void *obj, auto... args) {                                   \
        try {                                                                  \
            return reinterpret_cast<type *>(obj)->func(args...);               \
        } catch (...) {                                                        \
            return default_return;                                             \
        }                                                                      \
    }

#define UMA_ASSIGN_OP_NORETURN(ops, type, func)                                \
    ops.func = [](void *obj, auto... args) {                                   \
        try {                                                                  \
            return reinterpret_cast<type *>(obj)->func(args...);               \
        } catch (...) {                                                        \
        }                                                                      \
    }

/// @brief creates UMA memory provider based on given T type.
/// T should implement all functions defined by
/// uma_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize().
template <typename T, typename... Args>
auto memoryProviderMakeUnique(Args &&...args) {
    uma_memory_provider_ops_t ops;
    auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

    ops.version = UMA_VERSION_CURRENT;
    ops.initialize = [](void *params, void **obj) {
        auto *tuple = reinterpret_cast<decltype(argsTuple) *>(params);
        T *provider;
        try {
            provider = new T;
        } catch (...) {
            return UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

        *obj = provider;

        try {
            auto ret =
                std::apply(&T::initialize,
                           std::tuple_cat(std::make_tuple(provider), *tuple));
            if (ret != UMA_RESULT_SUCCESS) {
                delete provider;
            }
            return ret;
        } catch (...) {
            delete provider;
            return UMA_RESULT_ERROR_UNKNOWN;
        }
    };
    ops.finalize = [](void *obj) { delete reinterpret_cast<T *>(obj); };

    UMA_ASSIGN_OP(ops, T, alloc, UMA_RESULT_ERROR_UNKNOWN);
    UMA_ASSIGN_OP(ops, T, free, UMA_RESULT_ERROR_UNKNOWN);
    UMA_ASSIGN_OP_NORETURN(ops, T, get_last_native_error);
    UMA_ASSIGN_OP(ops, T, get_recommended_page_size, UMA_RESULT_ERROR_UNKNOWN);
    UMA_ASSIGN_OP(ops, T, get_min_page_size, UMA_RESULT_ERROR_UNKNOWN);
    UMA_ASSIGN_OP(ops, T, purge_lazy, UMA_RESULT_ERROR_UNKNOWN);
    UMA_ASSIGN_OP(ops, T, purge_force, UMA_RESULT_ERROR_UNKNOWN);
    UMA_ASSIGN_OP_NORETURN(ops, T, get_name);

    uma_memory_provider_handle_t hProvider = nullptr;
    auto ret = umaMemoryProviderCreate(&ops, &argsTuple, &hProvider);
    return std::pair<uma_result_t, provider_unique_handle_t>{
        ret, provider_unique_handle_t(hProvider, &umaMemoryProviderDestroy)};
}

/// @brief creates UMA memory pool based on given T type.
/// T should implement all functions defined by
/// uma_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize().
template <typename T, typename... Args>
auto poolMakeUnique(uma_memory_provider_handle_t *providers,
                    size_t numProviders, Args &&...args) {
    uma_memory_pool_ops_t ops;
    auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

    ops.version = UMA_VERSION_CURRENT;
    ops.initialize = [](uma_memory_provider_handle_t *providers,
                        size_t numProviders, void *params, void **obj) {
        auto *tuple = reinterpret_cast<decltype(argsTuple) *>(params);
        T *pool;

        try {
            pool = new T;
        } catch (...) {
            return UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

        *obj = pool;

        try {
            auto ret = std::apply(
                &T::initialize,
                std::tuple_cat(std::make_tuple(pool, providers, numProviders),
                               *tuple));
            if (ret != UMA_RESULT_SUCCESS) {
                delete pool;
            }
            return ret;
        } catch (...) {
            delete pool;
            return UMA_RESULT_ERROR_UNKNOWN;
        }
    };
    ops.finalize = [](void *obj) { delete reinterpret_cast<T *>(obj); };

    UMA_ASSIGN_OP(ops, T, malloc, ((void *)nullptr));
    UMA_ASSIGN_OP(ops, T, calloc, ((void *)nullptr));
    UMA_ASSIGN_OP(ops, T, aligned_malloc, ((void *)nullptr));
    UMA_ASSIGN_OP(ops, T, realloc, ((void *)nullptr));
    UMA_ASSIGN_OP(ops, T, malloc_usable_size, ((size_t)0));
    UMA_ASSIGN_OP_NORETURN(ops, T, free);
    UMA_ASSIGN_OP(ops, T, get_last_allocation_error, UMA_RESULT_ERROR_UNKNOWN);

    uma_memory_pool_handle_t hPool = nullptr;
    auto ret = umaPoolCreate(&ops, providers, numProviders, &argsTuple, &hPool);
    return std::pair<uma_result_t, pool_unique_handle_t>{
        ret, pool_unique_handle_t(hPool, &umaPoolDestroy)};
}

} // namespace uma

#endif /* UMA_HELPERS_H */

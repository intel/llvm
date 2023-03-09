/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UMA_HELPERS_H
#define UMA_HELPERS_H 1

#include <uma/memory_pool.h>
#include <uma/memory_pool_ops.h>
#include <uma/memory_provider.h>
#include <uma/memory_provider_ops.h>

#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace uma {

using pool_unique_handle_t = std::unique_ptr<uma_memory_pool_t, decltype(&umaPoolDestroy)>;
using provider_unique_handle_t = std::unique_ptr<uma_memory_provider_t, decltype(&umaMemoryProviderDestroy)>;

/// @brief creates UMA memory provider based on given T type.
/// T should implement all functions defined by
/// uma_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize(). All functions of T
/// should be noexcept.
template <typename T, typename... Args>
auto memoryProviderMakeUnique(Args &&...args) {
    uma_memory_provider_ops_t ops;
    auto argsTuple = std::make_tuple(std::forward<Args>(args)...);
    static_assert(noexcept(std::declval<T>().initialize(std::forward<Args>(args)...)));

    ops.version = UMA_VERSION_CURRENT;
    ops.initialize = [](void *params, void **obj) {
        auto *tuple = reinterpret_cast<decltype(argsTuple) *>(params);
        auto provider = new T;
        *obj = provider;
        return std::apply(&T::initialize, std::tuple_cat(std::make_tuple(*provider), *tuple));
    };
    ops.finalize = [](void *obj) {
        delete reinterpret_cast<T *>(obj);
    };
    ops.alloc = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->alloc(args...)));
        return reinterpret_cast<T *>(obj)->alloc(args...);
    };
    ops.free = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->free(args...)));
        return reinterpret_cast<T *>(obj)->free(args...);
    };
    ops.get_last_result = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->get_last_result(args...)));
        return reinterpret_cast<T *>(obj)->get_last_result(args...);
    };

    uma_memory_provider_handle_t hProvider = nullptr;
    auto ret = umaMemoryProviderCreate(&ops, &argsTuple, &hProvider);
    return std::pair<uma_result_t, provider_unique_handle_t>{ret, provider_unique_handle_t(hProvider, &umaMemoryProviderDestroy)};
}

/// @brief creates UMA memory pool based on given T type.
/// T should implement all functions defined by
/// uma_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize(). All functions of T
/// should be noexcept.
template <typename T, typename... Args>
auto poolMakeUnique(Args &&...args) {
    uma_memory_pool_ops_t ops;
    auto argsTuple = std::make_tuple(std::forward<Args>(args)...);
    static_assert(noexcept(std::declval<T>().initialize(std::forward<Args>(args)...)));

    ops.version = UMA_VERSION_CURRENT;
    ops.initialize = [](void *params, void **obj) {
        auto *tuple = reinterpret_cast<decltype(argsTuple) *>(params);
        auto pool = new T;
        *obj = pool;
        return std::apply(&T::initialize, std::tuple_cat(std::make_tuple(*pool), *tuple));
    };
    ops.finalize = [](void *obj) {
        delete reinterpret_cast<T *>(obj);
    };
    ops.malloc = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->malloc(args...)));
        return reinterpret_cast<T *>(obj)->malloc(args...);
    };
    ops.aligned_malloc = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->aligned_malloc(args...)));
        return reinterpret_cast<T *>(obj)->aligned_malloc(args...);
    };
    ops.realloc = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->realloc(args...)));
        return reinterpret_cast<T *>(obj)->realloc(args...);
    };
    ops.malloc_usable_size = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->malloc_usable_size(args...)));
        return reinterpret_cast<T *>(obj)->malloc_usable_size(args...);
    };
    ops.free = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->free(args...)));
        reinterpret_cast<T *>(obj)->free(args...);
    };
    ops.get_last_result = [](void *obj, auto... args) {
        static_assert(noexcept(reinterpret_cast<T *>(obj)->get_last_result(args...)));
        return reinterpret_cast<T *>(obj)->get_last_result(args...);
    };

    uma_memory_pool_handle_t hPool = nullptr;
    auto ret = umaPoolCreate(&ops, &argsTuple, &hPool);
    return std::pair<uma_result_t, pool_unique_handle_t>{ret, pool_unique_handle_t(hPool, &umaPoolDestroy)};
}
} // namespace uma

#endif /* UMA_HELPERS_H */

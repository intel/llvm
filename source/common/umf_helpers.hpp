/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_HELPERS_H
#define UMF_HELPERS_H 1

#include <umf/memory_pool.h>
#include <umf/memory_pool_ops.h>
#include <umf/memory_provider.h>
#include <umf/memory_provider_ops.h>
#include <ur_api.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace umf {

using pool_unique_handle_t =
    std::unique_ptr<umf_memory_pool_t,
                    std::function<void(umf_memory_pool_handle_t)>>;
using provider_unique_handle_t =
    std::unique_ptr<umf_memory_provider_t,
                    std::function<void(umf_memory_provider_handle_t)>>;

#define UMF_ASSIGN_OP(ops, type, func, default_return)                         \
    ops.func = [](void *obj, auto... args) {                                   \
        try {                                                                  \
            return reinterpret_cast<type *>(obj)->func(args...);               \
        } catch (...) {                                                        \
            return default_return;                                             \
        }                                                                      \
    }

#define UMF_ASSIGN_OP_NORETURN(ops, type, func)                                \
    ops.func = [](void *obj, auto... args) {                                   \
        try {                                                                  \
            return reinterpret_cast<type *>(obj)->func(args...);               \
        } catch (...) {                                                        \
        }                                                                      \
    }

/// @brief creates UMF memory provider based on given T type.
/// T should implement all functions defined by
/// umf_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize().
template <typename T, typename... Args>
auto memoryProviderMakeUnique(Args &&...args) {
    umf_memory_provider_ops_t ops;
    auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

    ops.version = UMF_VERSION_CURRENT;
    ops.initialize = [](void *params, void **obj) {
        auto *tuple = reinterpret_cast<decltype(argsTuple) *>(params);
        T *provider;
        try {
            provider = new T;
        } catch (...) {
            return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

        *obj = provider;

        try {
            auto ret =
                std::apply(&T::initialize,
                           std::tuple_cat(std::make_tuple(provider), *tuple));
            if (ret != UMF_RESULT_SUCCESS) {
                delete provider;
            }
            return ret;
        } catch (...) {
            delete provider;
            return UMF_RESULT_ERROR_UNKNOWN;
        }
    };
    ops.finalize = [](void *obj) { delete reinterpret_cast<T *>(obj); };

    UMF_ASSIGN_OP(ops, T, alloc, UMF_RESULT_ERROR_UNKNOWN);
    UMF_ASSIGN_OP(ops, T, free, UMF_RESULT_ERROR_UNKNOWN);
    UMF_ASSIGN_OP_NORETURN(ops, T, get_last_native_error);
    UMF_ASSIGN_OP(ops, T, get_recommended_page_size, UMF_RESULT_ERROR_UNKNOWN);
    UMF_ASSIGN_OP(ops, T, get_min_page_size, UMF_RESULT_ERROR_UNKNOWN);
    UMF_ASSIGN_OP(ops, T, purge_lazy, UMF_RESULT_ERROR_UNKNOWN);
    UMF_ASSIGN_OP(ops, T, purge_force, UMF_RESULT_ERROR_UNKNOWN);
    UMF_ASSIGN_OP(ops, T, get_name, "");

    umf_memory_provider_handle_t hProvider = nullptr;
    auto ret = umfMemoryProviderCreate(&ops, &argsTuple, &hProvider);
    return std::pair<umf_result_t, provider_unique_handle_t>{
        ret, provider_unique_handle_t(hProvider, &umfMemoryProviderDestroy)};
}

/// @brief creates UMF memory pool based on given T type.
/// T should implement all functions defined by
/// umf_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize().
template <typename T, typename... Args>
auto poolMakeUnique(umf_memory_provider_handle_t *providers,
                    size_t numProviders, Args &&...args) {
    umf_memory_pool_ops_t ops;
    auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

    ops.version = UMF_VERSION_CURRENT;
    ops.initialize = [](umf_memory_provider_handle_t *providers,
                        size_t numProviders, void *params, void **obj) {
        auto *tuple = reinterpret_cast<decltype(argsTuple) *>(params);
        T *pool;

        try {
            pool = new T;
        } catch (...) {
            return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }

        *obj = pool;

        try {
            auto ret = std::apply(
                &T::initialize,
                std::tuple_cat(std::make_tuple(pool, providers, numProviders),
                               *tuple));
            if (ret != UMF_RESULT_SUCCESS) {
                delete pool;
            }
            return ret;
        } catch (...) {
            delete pool;
            return UMF_RESULT_ERROR_UNKNOWN;
        }
    };
    ops.finalize = [](void *obj) { delete reinterpret_cast<T *>(obj); };

    UMF_ASSIGN_OP(ops, T, malloc, ((void *)nullptr));
    UMF_ASSIGN_OP(ops, T, calloc, ((void *)nullptr));
    UMF_ASSIGN_OP(ops, T, aligned_malloc, ((void *)nullptr));
    UMF_ASSIGN_OP(ops, T, realloc, ((void *)nullptr));
    UMF_ASSIGN_OP(ops, T, malloc_usable_size, ((size_t)0));
    UMF_ASSIGN_OP_NORETURN(ops, T, free);
    UMF_ASSIGN_OP(ops, T, get_last_allocation_error, UMF_RESULT_ERROR_UNKNOWN);

    umf_memory_pool_handle_t hPool = nullptr;
    auto ret = umfPoolCreate(&ops, providers, numProviders, &argsTuple, &hPool);
    return std::pair<umf_result_t, pool_unique_handle_t>{
        ret, pool_unique_handle_t(hPool, &umfPoolDestroy)};
}

template <typename Type> umf_result_t &getPoolLastStatusRef() {
    static thread_local umf_result_t last_status = UMF_RESULT_SUCCESS;
    return last_status;
}

/// @brief translates UMF return values to UR.
/// This function assumes that the native error of
/// the last failed memory provider is ur_result_t.
inline ur_result_t umf2urResult(umf_result_t umfResult) {
    switch (umfResult) {
    case UMF_RESULT_SUCCESS:
        return UR_RESULT_SUCCESS;
    case UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    case UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC: {
        auto hProvider = umfGetLastFailedMemoryProvider();
        if (hProvider == nullptr) {
            return UR_RESULT_ERROR_UNKNOWN;
        }

        ur_result_t Err = UR_RESULT_ERROR_UNKNOWN;
        umfMemoryProviderGetLastNativeError(hProvider, nullptr,
                                            reinterpret_cast<int32_t *>(&Err));
        return Err;
    }
    case UMF_RESULT_ERROR_INVALID_ARGUMENT:
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    case UMF_RESULT_ERROR_INVALID_ALIGNMENT:
        return UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT;
    case UMF_RESULT_ERROR_NOT_SUPPORTED:
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    default:
        return UR_RESULT_ERROR_UNKNOWN;
    };
}

} // namespace umf

#endif /* UMF_HELPERS_H */

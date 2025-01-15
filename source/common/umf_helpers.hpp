/*
 *
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
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

#include "logger/ur_logger.hpp"

#include <array>
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

#define DEFINE_CHECK_OP(op)                                                    \
  template <typename T> class HAS_OP_##op {                                    \
    typedef char check_success;                                                \
    typedef long check_fail;                                                   \
    template <typename U> static check_success test(decltype(&U::op));         \
    template <typename U> static check_fail test(...);                         \
                                                                               \
  public:                                                                      \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(check_success); \
  };                                                                           \
                                                                               \
  template <typename T, typename... Args>                                      \
  static inline                                                                \
      typename std::enable_if<HAS_OP_##op<T>::value, umf_result_t>::type       \
          CALL_OP_##op(T *t, Args &&...args) {                                 \
    return t->op(std::forward<Args>(args)...);                                 \
  }                                                                            \
                                                                               \
  static inline umf_result_t CALL_OP_##op(...) {                               \
    return UMF_RESULT_ERROR_NOT_SUPPORTED;                                     \
  }

DEFINE_CHECK_OP(get_ipc_handle_size)
DEFINE_CHECK_OP(get_ipc_handle)
DEFINE_CHECK_OP(put_ipc_handle)
DEFINE_CHECK_OP(open_ipc_handle)
DEFINE_CHECK_OP(close_ipc_handle)

#define UMF_ASSIGN_OP(ops, type, func, default_return)                         \
  ops.func = [](void *obj, auto... args) {                                     \
    try {                                                                      \
      return reinterpret_cast<type *>(obj)->func(args...);                     \
    } catch (...) {                                                            \
      return default_return;                                                   \
    }                                                                          \
  }

#define UMF_ASSIGN_OP_NORETURN(ops, type, func)                                \
  ops.func = [](void *obj, auto... args) {                                     \
    try {                                                                      \
      return reinterpret_cast<type *>(obj)->func(args...);                     \
    } catch (...) {                                                            \
    }                                                                          \
  }

#define UMF_ASSIGN_OP_OPT(ops, type, func, default_return)                     \
  ops.func = [](void *obj, auto... args) {                                     \
    try {                                                                      \
      return CALL_OP_##func(reinterpret_cast<type *>(obj), args...);           \
    } catch (...) {                                                            \
      return default_return;                                                   \
    }                                                                          \
  }

namespace detail {
template <typename T, typename ArgsTuple>
umf_result_t initialize(T *obj, ArgsTuple &&args) {
  try {
    auto ret = std::apply(
        &T::initialize,
        std::tuple_cat(std::make_tuple(obj), std::forward<ArgsTuple>(args)));
    if (ret != UMF_RESULT_SUCCESS) {
      delete obj;
    }
    return ret;
  } catch (...) {
    delete obj;
    return UMF_RESULT_ERROR_UNKNOWN;
  }
}

template <typename T, typename ArgsTuple>
umf_memory_pool_ops_t poolMakeUniqueOps() {
  umf_memory_pool_ops_t ops = {};

  ops.version = UMF_VERSION_CURRENT;
  ops.initialize = [](umf_memory_provider_handle_t provider, void *params,
                      void **obj) {
    try {
      *obj = new T;
    } catch (...) {
      return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return detail::initialize<T>(
        reinterpret_cast<T *>(*obj),
        std::tuple_cat(std::make_tuple(provider),
                       *reinterpret_cast<ArgsTuple *>(params)));
  };
  ops.finalize = [](void *obj) { delete reinterpret_cast<T *>(obj); };

  UMF_ASSIGN_OP(ops, T, malloc, ((void *)nullptr));
  UMF_ASSIGN_OP(ops, T, calloc, ((void *)nullptr));
  UMF_ASSIGN_OP(ops, T, aligned_malloc, ((void *)nullptr));
  UMF_ASSIGN_OP(ops, T, realloc, ((void *)nullptr));
  UMF_ASSIGN_OP(ops, T, malloc_usable_size, ((size_t)0));
  UMF_ASSIGN_OP(ops, T, free, UMF_RESULT_SUCCESS);
  UMF_ASSIGN_OP(ops, T, get_last_allocation_error, UMF_RESULT_ERROR_UNKNOWN);

  return ops;
}
} // namespace detail

/// @brief creates UMF memory provider based on given T type.
/// T should implement all functions defined by
/// umf_memory_provider_ops_t, except for finalize (it is
/// replaced by dtor). All arguments passed to this function are
/// forwarded to T::initialize().
template <typename T, typename... Args>
auto memoryProviderMakeUnique(Args &&...args) {
  umf_memory_provider_ops_t ops = {};
  auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

  ops.version = UMF_VERSION_CURRENT;
  ops.initialize = [](void *params, void **obj) {
    try {
      *obj = new T;
    } catch (...) {
      return UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    return detail::initialize<T>(
        reinterpret_cast<T *>(*obj),
        *reinterpret_cast<decltype(argsTuple) *>(params));
  };
  ops.finalize = [](void *obj) { delete reinterpret_cast<T *>(obj); };

  UMF_ASSIGN_OP(ops, T, alloc, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP_NORETURN(ops, T, get_last_native_error);
  UMF_ASSIGN_OP(ops, T, get_recommended_page_size, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP(ops, T, get_min_page_size, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP(ops, T, get_name, "");
  UMF_ASSIGN_OP(ops.ext, T, free, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP(ops.ext, T, purge_lazy, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP(ops.ext, T, purge_force, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP(ops.ext, T, allocation_merge, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP(ops.ext, T, allocation_split, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP_OPT(ops.ipc, T, get_ipc_handle_size, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP_OPT(ops.ipc, T, get_ipc_handle, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP_OPT(ops.ipc, T, put_ipc_handle, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP_OPT(ops.ipc, T, open_ipc_handle, UMF_RESULT_ERROR_UNKNOWN);
  UMF_ASSIGN_OP_OPT(ops.ipc, T, close_ipc_handle, UMF_RESULT_ERROR_UNKNOWN);

  umf_memory_provider_handle_t hProvider = nullptr;
  auto ret = umfMemoryProviderCreate(&ops, &argsTuple, &hProvider);
  return std::pair<umf_result_t, provider_unique_handle_t>{
      ret, provider_unique_handle_t(hProvider, &umfMemoryProviderDestroy)};
}

/// @brief creates UMF memory pool based on given T type.
/// This overload takes ownership of memory providers and destroys
/// them after memory pool is destroyed.
template <typename T, typename... Args>
auto poolMakeUnique(provider_unique_handle_t provider, Args &&...args) {
  auto argsTuple = std::make_tuple(std::forward<Args>(args)...);
  auto ops = detail::poolMakeUniqueOps<T, decltype(argsTuple)>();

  umf_memory_pool_handle_t hPool = nullptr;

  auto ret = umfPoolCreate(&ops, provider.get(), &argsTuple,
                           UMF_POOL_CREATE_FLAG_OWN_PROVIDER, &hPool);
  if (ret == UMF_RESULT_SUCCESS) {
    provider.release(); // pool now owns the provider
  }
  return std::pair<umf_result_t, pool_unique_handle_t>{
      ret, pool_unique_handle_t(hPool, umfPoolDestroy)};
}

static inline auto poolMakeUniqueFromOps(umf_memory_pool_ops_t *ops,
                                         provider_unique_handle_t provider,
                                         void *params) {
  umf_memory_pool_handle_t hPool;
  auto ret = umfPoolCreate(ops, provider.get(), params,
                           UMF_POOL_CREATE_FLAG_OWN_PROVIDER, &hPool);
  if (ret != UMF_RESULT_SUCCESS) {
    return std::pair<umf_result_t, pool_unique_handle_t>{
        ret, pool_unique_handle_t(nullptr, nullptr)};
  }

  provider.release(); // pool now owns the provider

  return std::pair<umf_result_t, pool_unique_handle_t>{
      UMF_RESULT_SUCCESS, pool_unique_handle_t(hPool, umfPoolDestroy)};
}

static inline auto providerMakeUniqueFromOps(umf_memory_provider_ops_t *ops,
                                             void *params) {
  umf_memory_provider_handle_t hProvider;
  auto ret = umfMemoryProviderCreate(ops, params, &hProvider);
  if (ret != UMF_RESULT_SUCCESS) {
    return std::pair<umf_result_t, provider_unique_handle_t>{
        ret, provider_unique_handle_t(nullptr, nullptr)};
  }

  return std::pair<umf_result_t, provider_unique_handle_t>{
      UMF_RESULT_SUCCESS,
      provider_unique_handle_t(hProvider, umfMemoryProviderDestroy)};
}

template <typename Type> umf_result_t &getPoolLastStatusRef() {
  static thread_local umf_result_t last_status = UMF_RESULT_SUCCESS;
  return last_status;
}

ur_result_t getProviderNativeError(const char *providerName,
                                   int32_t nativeError);

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

    int32_t Err = UR_RESULT_ERROR_UNKNOWN;
    const char *Msg = nullptr;
    umfMemoryProviderGetLastNativeError(hProvider, &Msg, &Err);

    if (Msg) {
      logger::error("UMF failed with: {}", Msg);
    }

    return getProviderNativeError(umfMemoryProviderGetName(hProvider), Err);
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

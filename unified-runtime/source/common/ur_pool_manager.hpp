/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef USM_POOL_MANAGER_HPP
#define USM_POOL_MANAGER_HPP 1

#include <ur_ddi.h>

#include "logger/ur_logger.hpp"
#include "umf_helpers.hpp"
#include "ur_api.h"
#include "ur_util.hpp"

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/pools/pool_disjoint.h>
#include <umf/pools/pool_proxy.h>

#include <functional>
#include <unordered_map>
#include <vector>

namespace usm {

namespace detail {
struct ddiTables {
  ddiTables() {
    auto ret =
        urGetDeviceProcAddrTable(UR_API_VERSION_CURRENT, &deviceDdiTable);
    if (ret != UR_RESULT_SUCCESS) {
      throw ret;
    }

    ret = urGetContextProcAddrTable(UR_API_VERSION_CURRENT, &contextDdiTable);
    if (ret != UR_RESULT_SUCCESS) {
      throw ret;
    }
  }
  ur_device_dditable_t deviceDdiTable;
  ur_context_dditable_t contextDdiTable;
};
} // namespace detail

/// @brief describes an internal USM pool instance.
struct pool_descriptor {
  ur_usm_pool_handle_t poolHandle;

  ur_context_handle_t hContext;
  ur_device_handle_t hDevice;
  ur_usm_type_t type;
  bool deviceReadOnly;

  bool operator==(const pool_descriptor &other) const;
  friend std::ostream &operator<<(std::ostream &os,
                                  const pool_descriptor &desc);
  static std::vector<pool_descriptor>
  createFromDevices(ur_usm_pool_handle_t poolHandle,
                    ur_context_handle_t hContext,
                    const std::vector<ur_device_handle_t> &devices);
};

static inline bool
isSharedAllocationReadOnlyOnDevice(const pool_descriptor &desc) {
  return desc.type == UR_USM_TYPE_SHARED && desc.deviceReadOnly;
}

inline bool pool_descriptor::operator==(const pool_descriptor &other) const {
  static usm::detail::ddiTables ddi;

  const pool_descriptor &lhs = *this;
  const pool_descriptor &rhs = other;
  ur_native_handle_t lhsNative = 0, rhsNative = 0;

  // We want to share a memory pool for sub-devices and sub-sub devices.
  // Sub-devices and sub-sub-devices might be represented by different
  // ur_device_handle_t but they share the same native_handle_t (which is used
  // by UMF provider). Ref:
  // https://github.com/intel/llvm/commit/86511c5dc84b5781dcfd828caadcb5cac157eae1
  // TODO: is this L0 specific?
  if (lhs.hDevice) {
    auto ret = ddi.deviceDdiTable.pfnGetNativeHandle(lhs.hDevice, &lhsNative);
    if (ret != UR_RESULT_SUCCESS) {
      throw ret;
    }
  }

  if (rhs.hDevice) {
    auto ret = ddi.deviceDdiTable.pfnGetNativeHandle(rhs.hDevice, &rhsNative);
    if (ret != UR_RESULT_SUCCESS) {
      throw ret;
    }
  }

  return lhsNative == rhsNative && lhs.type == rhs.type &&
         (isSharedAllocationReadOnlyOnDevice(lhs) ==
          isSharedAllocationReadOnlyOnDevice(rhs)) &&
         lhs.poolHandle == rhs.poolHandle;
}

inline std::ostream &operator<<(std::ostream &os, const pool_descriptor &desc) {
  os << "pool handle: " << desc.poolHandle
     << " context handle: " << desc.hContext
     << " device handle: " << desc.hDevice << " memory type: " << desc.type
     << " is read only: " << desc.deviceReadOnly;
  return os;
}

inline std::vector<pool_descriptor> pool_descriptor::createFromDevices(
    ur_usm_pool_handle_t poolHandle, ur_context_handle_t hContext,
    const std::vector<ur_device_handle_t> &devices) {
  std::vector<pool_descriptor> descriptors;
  pool_descriptor &desc = descriptors.emplace_back();
  desc.poolHandle = poolHandle;
  desc.hContext = hContext;
  desc.type = UR_USM_TYPE_HOST;

  for (auto &device : devices) {
    {
      pool_descriptor &desc = descriptors.emplace_back();
      desc.poolHandle = poolHandle;
      desc.hContext = hContext;
      desc.hDevice = device;
      desc.type = UR_USM_TYPE_DEVICE;
    }
    {
      pool_descriptor &desc = descriptors.emplace_back();
      desc.poolHandle = poolHandle;
      desc.hContext = hContext;
      desc.type = UR_USM_TYPE_SHARED;
      desc.hDevice = device;
      desc.deviceReadOnly = false;
    }
    {
      pool_descriptor &desc = descriptors.emplace_back();
      desc.poolHandle = poolHandle;
      desc.hContext = hContext;
      desc.type = UR_USM_TYPE_SHARED;
      desc.hDevice = device;
      desc.deviceReadOnly = true;
    }
  }

  return descriptors;
}

template <typename D, typename H> struct pool_manager {
private:
  using pool_handle_t = H *;
  using unique_pool_handle_t = std::unique_ptr<H, std::function<void(H *)>>;
  using desc_to_pool_map_t = std::unordered_map<D, unique_pool_handle_t>;

  desc_to_pool_map_t descToPoolMap;

public:
  static std::pair<ur_result_t, pool_manager>
  create(desc_to_pool_map_t &&descToHandleMap = {}) {
    auto manager = pool_manager();

    for (auto &[desc, hPool] : descToHandleMap) {
      auto ret = manager.addPool(desc, std::move(hPool));
      if (ret != UR_RESULT_SUCCESS) {
        return {ret, pool_manager()};
      }
    }

    return {UR_RESULT_SUCCESS, std::move(manager)};
  }

  ur_result_t addPool(const D &desc, unique_pool_handle_t &&hPool) {
    if (!descToPoolMap.try_emplace(desc, std::move(hPool)).second) {
      UR_LOG(ERR, "Pool for pool descriptor: {}, already exists", desc);
      return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return UR_RESULT_SUCCESS;
  }

  std::optional<pool_handle_t> getPool(const D &desc) {
    auto it = descToPoolMap.find(desc);
    if (it == descToPoolMap.end()) {
      UR_LOG(ERR, "Pool descriptor doesn't match any existing pool: {}", desc);
      return std::nullopt;
    }

    return it->second.get();
  }
  template <typename Func> void forEachPool(Func func) {
    for (const auto &[desc, pool] : descToPoolMap) {
      if (!func(pool.get()))
        break;
    }
  }
};

inline umf::pool_unique_handle_t
makeDisjointPool(umf::provider_unique_handle_t &&provider,
                 usm::umf_disjoint_pool_config_t &poolParams) {
  auto umfParams = getUmfParamsHandle(poolParams);
  auto [ret, poolHandle] =
      umf::poolMakeUniqueFromOps(umfDisjointPoolOps(), std::move(provider),
                                 static_cast<void *>(umfParams.get()));
  if (ret != UMF_RESULT_SUCCESS)
    throw umf::umf2urResult(ret);
  return std::move(poolHandle);
}

inline umf::pool_unique_handle_t
makeProxyPool(umf::provider_unique_handle_t &&provider) {
  auto [ret, poolHandle] = umf::poolMakeUniqueFromOps(
      umfProxyPoolOps(), std::move(provider), nullptr);
  if (ret != UMF_RESULT_SUCCESS)
    throw umf::umf2urResult(ret);

  return std::move(poolHandle);
}

} // namespace usm

namespace std {
/// @brief hash specialization for usm::pool_descriptor
template <> struct hash<usm::pool_descriptor> {
  inline size_t operator()(const usm::pool_descriptor &desc) const {
    static usm::detail::ddiTables ddi;

    ur_native_handle_t native = 0;
    if (desc.hDevice) {
      auto ret = ddi.deviceDdiTable.pfnGetNativeHandle(desc.hDevice, &native);
      if (ret != UR_RESULT_SUCCESS) {
        throw ret;
      }
    }

    return combine_hashes(0, desc.type, native,
                          isSharedAllocationReadOnlyOnDevice(desc),
                          desc.poolHandle);
  }
};

} // namespace std

#endif /* USM_POOL_MANAGER_HPP */

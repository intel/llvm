//===--- disjoint_pool_config_parser.hpp -configuration for USM memory pool-==//
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef USM_POOL_CONFIG
#define USM_POOL_CONFIG

#include <umf/pools/pool_disjoint.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace usm {
enum DisjointPoolMemType { Host, Device, Shared, SharedReadOnly, All };

typedef struct umf_disjoint_pool_config_t {
    umf_disjoint_pool_config_t();

    size_t SlabMinSize;
    size_t MaxPoolableSize;
    size_t Capacity;
    size_t MinBucketSize;
    int PoolTrace;
    umf_disjoint_pool_shared_limits_t *SharedLimits;
    const char *Name;
} umf_disjoint_pool_config_t;

using umfDisjointPoolParamsHandle =
    std::unique_ptr<umf_disjoint_pool_params_t,
                    decltype(&umfDisjointPoolParamsDestroy)>;

// Stores configuration for all instances of USM allocator
class DisjointPoolAllConfigs {
  public:
    size_t EnableBuffers = 1;
    std::shared_ptr<umf_disjoint_pool_shared_limits_t> limits;
    umf_disjoint_pool_config_t Configs[DisjointPoolMemType::All];

    DisjointPoolAllConfigs(int trace = 0);
};

// Parse optional config parameters of this form:
// [EnableBuffers][;[MaxPoolSize][;memtypelimits]...]
//  memtypelimits: [<memtype>:]<limits>
//  memtype: host|device|shared
//  limits:  [MaxPoolableSize][,[Capacity][,SlabMinSize]]
//
// Without a memory type, the limits are applied to each memory type.
// Parameters are for each context, except MaxPoolSize, which is overall
// pool size for all contexts.
// Duplicate specifications will result in the right-most taking effect.
//
// EnableBuffers:   Apply chunking/pooling to SYCL buffers.
//                  Default 1.
// MaxPoolSize:     Limit on overall unfreed memory.
//                  Default 16MB.
// MaxPoolableSize: Maximum allocation size subject to chunking/pooling.
//                  Default 2MB host, 4MB device and 0 shared.
// Capacity:        Maximum number of unfreed allocations in each bucket.
//                  Default 4.
// SlabMinSize:     Minimum allocation size requested from USM.
//                  Default 64KB host and device, 2MB shared.
//
// Example of usage:
// "1;32M;host:1M,4,64K;device:1M,4,64K;shared:0,0,2M"
DisjointPoolAllConfigs parseDisjointPoolConfig(const std::string &config,
                                               int trace = 1);

static inline void UMF_CALL_THROWS(umf_result_t res) {
    if (res != UMF_RESULT_SUCCESS) {
        throw res;
    }
}

static inline umfDisjointPoolParamsHandle
getUmfParamsHandle(umf_disjoint_pool_config_t &config) {
    umf_disjoint_pool_params_handle_t cHandle;
    UMF_CALL_THROWS(umfDisjointPoolParamsCreate(&cHandle));

    umfDisjointPoolParamsHandle handle(cHandle, &umfDisjointPoolParamsDestroy);
    UMF_CALL_THROWS(
        umfDisjointPoolParamsSetSlabMinSize(cHandle, config.SlabMinSize));
    UMF_CALL_THROWS(umfDisjointPoolParamsSetMaxPoolableSize(
        cHandle, config.MaxPoolableSize));
    UMF_CALL_THROWS(umfDisjointPoolParamsSetCapacity(cHandle, config.Capacity));
    UMF_CALL_THROWS(
        umfDisjointPoolParamsSetMinBucketSize(cHandle, config.MinBucketSize));
    UMF_CALL_THROWS(
        umfDisjointPoolParamsSetSharedLimits(cHandle, config.SharedLimits));
    UMF_CALL_THROWS(umfDisjointPoolParamsSetName(cHandle, config.Name));
    UMF_CALL_THROWS(umfDisjointPoolParamsSetTrace(cHandle, config.PoolTrace));
    return handle;
}
} // namespace usm

#endif

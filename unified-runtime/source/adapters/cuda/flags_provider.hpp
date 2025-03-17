
#pragma once
#include "umf_helpers.hpp"
#include <umf/memory_pool.h>

template <typename AllocMixin, typename FlagTy = unsigned, size_t ReserveSz = 4>
class FlagsMemProviderWithDefaultBase {
  struct KV {
    FlagTy Flags;
    umf_memory_pool_t *Pool;
  };
  umf_memory_pool_t *DefaultPool = nullptr;
  umf_memory_provider_t *DefaultProvider = nullptr;
  // There are lots of ways we could store these mappings, but since there
  // aren't likely to be thousands of different flags combinations in any
  // reasonable application, it's probably fine to keep them in an array, and
  // then simply linear search that array. If that turns out to not be the case
  // *in practise*, then we might want to switch to something more onerous like
  // a std::map. A survey of existing accelerator driver APIs suggest that there
  // aren't more than about 4 or 5 different flag combinations possible in the
  // wild, so I think the linear scan is the most globally optimal solution here
  std::optional<std::vector<KV>> PoolsWithFlags{std::nullopt};

public:
  // Get a provider for the memory flags `F`. If one can't be found it will be
  // created and cached before returning. On failure `NULL` is returned, and `E`
  // is set to indicate the error.
  // If `F` is falsey, the default provider is given
  umf_memory_pool_t *getPoolForFlags(FlagTy F, ur_result_t &E) {
    DefaultPool = DefaultPool ? DefaultPool : static_cast<AllocMixin *>(this)->alloc(FlagTy{}, E);
    if (!F)
      return DefaultPool;
    if (!PoolsWithFlags.has_value()) {
      PoolsWithFlags.emplace();
      PoolsWithFlags->reserve(ReserveSz);
    }

    auto It = std::find_if(std::begin(*PoolsWithFlags), std::end(*PoolsWithFlags),
        [F](KV &P) { return P.Flags == F; });
    if (It != std::end(*PoolsWithFlags))
      return It->Pool;
    // A provider with the given flags doesn't exist, so create one and give
    // that back
    if (umf_memory_pool_t *P = static_cast<AllocMixin *>(this)->alloc(F, E)) {
      assert(!E);
      PoolsWithFlags->push_back(KV{F, P});
      return P;
    }
    return nullptr;
  }

  ~FlagsMemProviderWithDefaultBase() {
    if (PoolsWithFlags)
      for (KV &KV : *PoolsWithFlags)
        umfPoolDestroy(KV.Pool);
    if (DefaultPool)
      umfPoolDestroy(DefaultPool);
  }
};

class CUDAFlagsUMFVendor
    : public FlagsMemProviderWithDefaultBase<CUDAFlagsUMFVendor> {
  CUcontext Ctx;
  CUdevice Device;
  umf_usm_memory_type_t MemType = UMF_MEMORY_TYPE_UNKNOWN;

public:
  umf_memory_pool_t *alloc(unsigned Flags, ur_result_t &Err) {
    umf_cuda_memory_provider_params_t *Params = nullptr;
    if (umf_result_t E = umfCUDAMemoryProviderParamsCreate(&Params);
        E != UMF_RESULT_SUCCESS) {
      Err = umf::umf2urResult(E);
      return nullptr;
    }
    OnScopeExit Cleanup([&]() { umfCUDAMemoryProviderParamsDestroy(Params); });

    umf::setCUMemoryProviderParams(Params, Device, Ctx, MemType);


    if (Flags)
    if (umf_result_t E =
            umfCUDAMemoryProviderParamsSetAllocFlags(Params, Flags);
        E != UMF_RESULT_SUCCESS) {
      Err = umf::umf2urResult(E);
      return nullptr;
    }
    // create UMF CUDA memory provider for the host memory
    umf_memory_provider_t *Provider = nullptr;
    if (umf_result_t E = umfMemoryProviderCreate(umfCUDAMemoryProviderOps(),
                                                 Params, &Provider);
        E != UMF_RESULT_SUCCESS) {
      Err = umf::umf2urResult(E);
    }
    umf_memory_pool_t *Pool = nullptr;
    if (umf_result_t E =
            umfPoolCreate(umfProxyPoolOps(), Provider, nullptr,
                          UMF_POOL_CREATE_FLAG_OWN_PROVIDER, &Pool);
        E != UMF_RESULT_SUCCESS) {
      Err = umf::umf2urResult(E);
      umfMemoryProviderDestroy(Provider);
    }
    return Pool;
  }

public:
  CUDAFlagsUMFVendor(CUcontext Ctx, CUdevice Device,
                     umf_usm_memory_type_t MemType)
      : Ctx(Ctx), Device(Device), MemType(MemType) {}
};

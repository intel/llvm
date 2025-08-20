//==---------------- kernel_data.hpp - SYCL handler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/kernel_name_based_cache.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

class KernelData {
public:
  using KernelParamDescGetterT = detail::kernel_param_desc_t (*)(int);

  KernelData() = default;
  ~KernelData() = default;
  KernelData(const KernelData &) = default;
  KernelData(KernelData &&) = default;
  KernelData &operator=(const KernelData &) = default;
  KernelData &operator=(KernelData &&) = default;

  void *getKernelFuncPtr() const { return MKernelFuncPtr; }

  size_t getKernelNumArgs() const { return MKernelNumArgs; }

  KernelParamDescGetterT getKernelParamDescGetter() const {
    return MKernelParamDescGetter;
  }

  bool isESIMD() const { return MKernelIsESIMD; }

  bool hasSpecialCaptures() const { return MKernelHasSpecialCaptures; }

  KernelNameBasedCacheT *getKernelNameBasedCachePtr() const {
    return MKernelNameBasedCachePtr;
  }

  void setKernelNameBasedCachePtr(KernelNameBasedCacheT *Ptr) {
    MKernelNameBasedCachePtr = Ptr;
  }

  void setKernelInfo(void *KernelFuncPtr, int KernelNumArgs,
                     KernelParamDescGetterT KernelParamDescGetter,
                     bool KernelIsESIMD, bool KernelHasSpecialCaptures) {
    MKernelFuncPtr = KernelFuncPtr;
    MKernelNumArgs = KernelNumArgs;
    MKernelParamDescGetter = KernelParamDescGetter;
    MKernelIsESIMD = KernelIsESIMD;
    MKernelHasSpecialCaptures = KernelHasSpecialCaptures;
  }

private:
  // Store information about the kernel arguments.
  void *MKernelFuncPtr = nullptr;
  size_t MKernelNumArgs = 0;
  KernelParamDescGetterT MKernelParamDescGetter = nullptr;
  bool MKernelIsESIMD = false;
  bool MKernelHasSpecialCaptures = true;

  // A pointer to a kernel name based cache retrieved on the application side.
  KernelNameBasedCacheT *MKernelNameBasedCachePtr = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

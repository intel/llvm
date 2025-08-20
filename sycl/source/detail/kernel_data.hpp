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

  size_t getKernelNumArgs() const { return MDeviceKernelInfoPtr->NumParams; }

  KernelParamDescGetterT getKernelParamDescGetter() const {
    return MDeviceKernelInfoPtr->ParamDescGetter;
  }

  bool isESIMD() const { return MDeviceKernelInfoPtr->IsESIMD; }

  bool hasSpecialCaptures() const {
    return MDeviceKernelInfoPtr->HasSpecialCaptures;
  }

  DeviceKernelInfo *getDeviceKernelInfoPtr() const {
    return MDeviceKernelInfoPtr;
  }

  void setDeviceKernelInfoPtr(DeviceKernelInfo *Ptr) {
    MDeviceKernelInfoPtr = Ptr;
  }
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void setKernelInfo(void *KernelFuncPtr, int KernelNumArgs,
                     KernelParamDescGetterT KernelParamDescGetter,
                     bool KernelIsESIMD, bool KernelHasSpecialCaptures) {
    MKernelFuncPtr = KernelFuncPtr;
    MDeviceKernelInfoPtr->NumParams = KernelNumArgs;
    MDeviceKernelInfoPtr->ParamDescGetter = KernelParamDescGetter;
    MDeviceKernelInfoPtr->IsESIMD = KernelIsESIMD;
    MDeviceKernelInfoPtr->HasSpecialCaptures = KernelHasSpecialCaptures;
  }
#endif

  void setKernelInfo(void *KernelFuncPtr,
                     detail::DeviceKernelInfo *DeviceKernelInfoPtr) {
    MKernelFuncPtr = KernelFuncPtr;
    MDeviceKernelInfoPtr = DeviceKernelInfoPtr;
  }

  bool usesAssert() const { return MDeviceKernelInfoPtr->usesAssert(); }

private:
  // Store information about the kernel arguments.
  void *MKernelFuncPtr = nullptr;

  // A pointer to device kernel information. Cached on the application side in
  // headers or retrieved from program manager.
  DeviceKernelInfo *MDeviceKernelInfoPtr = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

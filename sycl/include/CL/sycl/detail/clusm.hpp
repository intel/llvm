//==---------------- clusm.hpp - SYCL USM for CL Utils ---------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/cl.h>
#include <CL/cl_usm_ext.h>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/os_util.hpp>

#include <map>
#include <mutex>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {
namespace usm {

class CLUSM {
public:
  CLUSM() = default;
  ~CLUSM() = default;

  void initExtensions(cl_context Context, cl_platform_id Platform);

  void *hostMemAlloc(cl_context Context, cl_mem_properties_intel *Properties,
                     size_t Size, cl_uint Alignment, cl_int *Errcode_ret);
  void *deviceMemAlloc(cl_context Context, cl_device_id Device,
                       cl_mem_properties_intel *Properties, size_t Size,
                       cl_uint Alignment, cl_int *Errcode_ret);
  void *sharedMemAlloc(cl_context Context, cl_device_id Device,
                       cl_mem_properties_intel *Properties, size_t Size,
                       cl_uint Alignment, cl_int *Errcode_ret);

  cl_int memFree(cl_context Context, const void *Ptr);

  cl_int getMemAllocInfoINTEL(cl_context Context, const void *Ptr,
                              cl_mem_info_intel Param_name,
                              size_t Param_value_size, void *Param_value,
                              size_t *Param_value_size_ret);

  cl_int setKernelExecInfo(cl_kernel Kernel, cl_kernel_exec_info Param_name,
                           size_t Param_value_size, const void *Param_value);

  cl_int setKernelIndirectUSMExecInfo(cl_command_queue Queue, cl_kernel Kernel);

  template <class T>
  cl_int writeParamToMemory(size_t Param_value_size, T Param,
                            size_t *Param_value_size_ret, T *Pointer) const;

  bool useCLUSM() { return mEnableCLUSM; }

  bool isInitialized() { return mInitialized; }

private:
  bool mEnableCLUSM = true;
  bool mInitialized = false;
  std::mutex mLock;

  struct SUSMAllocInfo {
    SUSMAllocInfo() = default;

    cl_unified_shared_memory_type_intel Type = CL_MEM_TYPE_UNKNOWN_INTEL;
    const void *BaseAddress = nullptr;
    size_t Size = 0;
    size_t Alignment = 0;
  };

  using CUSMAllocMap = std::map<const void *, SUSMAllocInfo>;
  using CUSMAllocVector = std::vector<const void *>;

  struct SUSMContextInfo {
    CUSMAllocMap AllocMap;

    CUSMAllocVector HostAllocVector;
    // TODO: Support multiple devices by mapping device-> vector?
    CUSMAllocVector DeviceAllocVector;
    CUSMAllocVector SharedAllocVector;
  };

  // TODO: Support multiple contexts by mapping context -> USMContextInfo?
  SUSMContextInfo mUSMContextInfo;

  struct SUSMKernelInfo {
    SUSMKernelInfo() = default;

    bool IndirectHostAccess = false;
    bool IndirectDeviceAccess = false;
    bool IndirectSharedAccess = false;

    std::vector<void *> SVMPtrs;
  };

  typedef std::map<cl_kernel, SUSMKernelInfo> CUSMKernelInfoMap;

  CUSMKernelInfoMap mUSMKernelInfoMap;
};

} // namespace usm

namespace cliext {
bool initializeExtensions(cl_context context, cl_platform_id platform);
} // namespace cliext

} // namespace detail
} // namespace sycl
} // namespace cl

__SYCL_EXPORTED extern std::map<cl_context, cl::sycl::detail::usm::CLUSM *>
    gCLUSM;
inline cl::sycl::detail::usm::CLUSM *GetCLUSM(cl_context ctxt) {
  if (!cl::sycl::detail::pi::piUseBackend(
          cl::sycl::detail::pi::PiBackend::SYCL_BE_PI_OPENCL)) {
    // Bail if we're not using a CL backend. CLUSM is not relevant.
    return nullptr;
  }

  cl::sycl::detail::usm::CLUSM &*retVal = gCLUSM[ctxt];;
  if (retVal == nullptr) {
    retVal = new cl::sycl::detail::usm::CLUSM();
  }
  return retVal;
}

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
  static bool Create(CLUSM *&pCLUSM);
  static void Delete(CLUSM *&pCLUSM);

  void initExtensions(cl_platform_id platform);

  void *hostMemAlloc(cl_context context, cl_mem_properties_intel *properties,
                     size_t size, cl_uint alignment, cl_int *errcode_ret);
  void *deviceMemAlloc(cl_context context, cl_device_id device,
                       cl_mem_properties_intel *properties, size_t size,
                       cl_uint alignment, cl_int *errcode_ret);
  void *sharedMemAlloc(cl_context context, cl_device_id device,
                       cl_mem_properties_intel *properties, size_t size,
                       cl_uint alignment, cl_int *errcode_ret);

  cl_int memFree(cl_context context, const void *ptr);

  cl_int getMemAllocInfoINTEL(cl_context context, const void *ptr,
                              cl_mem_info_intel param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret);

  cl_int setKernelExecInfo(cl_kernel kernel, cl_kernel_exec_info param_name,
                           size_t param_value_size, const void *param_value);

  cl_int setKernelIndirectUSMExecInfo(cl_command_queue queue, cl_kernel kernel);

  template <class T>
  cl_int writeParamToMemory(size_t param_value_size, T param,
                            size_t *param_value_size_ret, T *pointer) const;

  bool useCLUSM() { return mEnableCLUSM; }

  bool isInitialized() { return mInitialized; }

private:
  bool mEnableCLUSM = true;
  bool mInitialized = false;
  std::mutex mLock;

  CLUSM() = default;
  ~CLUSM() = default;

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
bool initializeExtensions(cl_platform_id platform);
} // namespace cliext

} // namespace detail
} // namespace sycl
} // namespace cl

__SYCL_EXPORTED extern cl::sycl::detail::usm::CLUSM *gCLUSM;
inline cl::sycl::detail::usm::CLUSM *GetCLUSM() {
  if (gCLUSM == nullptr) {
    cl::sycl::detail::usm::CLUSM::Create(gCLUSM);
  }

  cl::sycl::detail::usm::CLUSM *retVal = nullptr;
  if (cl::sycl::detail::pi::useBackend(
          cl::sycl::detail::pi::Backend::SYCL_BE_PI_OPENCL)) {
    retVal = gCLUSM;
  }
  return retVal;
}

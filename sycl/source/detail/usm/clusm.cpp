//==---------------- clusm.cpp - USM for CL Utils  -------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#include <CL/sycl/detail/clusm.hpp>

#include <algorithm>
#include <cassert>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <time.h> // strdate

cl::sycl::detail::usm::CLUSM *gCLUSM = nullptr;

namespace cl {
namespace sycl {
namespace detail {
namespace usm {

bool CLUSM::Create(CLUSM *&pCLUSM) {
  pCLUSM = new CLUSM();
  if (pCLUSM) {
    return true;
  }

  return false;
}

void CLUSM::Delete(CLUSM *&pCLUSM) {
  delete pCLUSM;
  pCLUSM = nullptr;
}

void CLUSM::initExtensions(cl_platform_id platform) {
  // If OpenCL supports the USM Extension, don't enable CLUSM.
  std::lock_guard<std::mutex> guard(mLock);

  if (!mInitialized) {
    mEnableCLUSM = !cliext::initializeExtensions(platform);
    mInitialized = true;
  }
}

void *CLUSM::hostMemAlloc(cl_context context,
                          cl_mem_properties_intel *properties, size_t size,
                          cl_uint alignment, cl_int *errcode_ret) {
  std::lock_guard<std::mutex> guard(mLock);
  void *ptr =
      clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 size, alignment);

  cl_int errorCode = CL_SUCCESS;

  if (ptr != nullptr) {
    // Record this allocation in the alloc map:
    SUSMAllocInfo &allocInfo = mUSMContextInfo.AllocMap[ptr];
    allocInfo.Type = CL_MEM_TYPE_HOST_INTEL;
    allocInfo.BaseAddress = ptr;
    allocInfo.Size = size;
    allocInfo.Alignment = alignment;

    mUSMContextInfo.HostAllocVector.push_back(ptr);
  } else {
    errorCode = CL_OUT_OF_HOST_MEMORY; // TODO: which error?
  }

  if (errcode_ret) {
    errcode_ret[0] = errorCode;
  }

  return ptr;
}

void *CLUSM::deviceMemAlloc(cl_context context, cl_device_id device,
                            cl_mem_properties_intel *properties, size_t size,
                            cl_uint alignment, cl_int *errcode_ret) {
  std::lock_guard<std::mutex> guard(mLock);
  void *ptr = clSVMAlloc(context, CL_MEM_READ_WRITE, size, alignment);

  cl_int errorCode = CL_SUCCESS;

  if (ptr != nullptr) {
    // Record this allocation in the alloc map:
    SUSMAllocInfo &allocInfo = mUSMContextInfo.AllocMap[ptr];
    allocInfo.Type = CL_MEM_TYPE_DEVICE_INTEL;
    allocInfo.BaseAddress = ptr;
    allocInfo.Size = size;
    allocInfo.Alignment = alignment;

    mUSMContextInfo.DeviceAllocVector.push_back(ptr);
  } else {
    errorCode = CL_OUT_OF_HOST_MEMORY; // TODO: which error?
  }

  if (errcode_ret) {
    errcode_ret[0] = errorCode;
  }

  return ptr;
}

void *CLUSM::sharedMemAlloc(cl_context context, cl_device_id device,
                            cl_mem_properties_intel *properties, size_t size,
                            cl_uint alignment, cl_int *errcode_ret) {
  std::lock_guard<std::mutex> guard(mLock);
  void *ptr =
      clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 size, alignment);

  cl_int errorCode = CL_SUCCESS;

  if (ptr != nullptr) {
    // Record this allocation in the alloc map:
    SUSMAllocInfo &allocInfo = mUSMContextInfo.AllocMap[ptr];
    allocInfo.Type = CL_MEM_TYPE_SHARED_INTEL;
    allocInfo.BaseAddress = ptr;
    allocInfo.Size = size;
    allocInfo.Alignment = alignment;

    mUSMContextInfo.SharedAllocVector.push_back(ptr);
  } else {
    errorCode = CL_OUT_OF_HOST_MEMORY; // TODO: which error?
  }

  if (errcode_ret) {
    errcode_ret[0] = errorCode;
  }

  return ptr;
}

cl_int CLUSM::memFree(cl_context context, const void *ptr) {
  std::lock_guard<std::mutex> guard(mLock);

  CUSMAllocMap::iterator iter = mUSMContextInfo.AllocMap.find(ptr);
  if (iter != mUSMContextInfo.AllocMap.end()) {
    const SUSMAllocInfo &allocInfo = iter->second;

    switch (allocInfo.Type) {
    case CL_MEM_TYPE_HOST_INTEL:
      mUSMContextInfo.HostAllocVector.erase(
          std::find(mUSMContextInfo.HostAllocVector.begin(),
                    mUSMContextInfo.HostAllocVector.end(), ptr));
      break;
    case CL_MEM_TYPE_DEVICE_INTEL:
      mUSMContextInfo.DeviceAllocVector.erase(
          std::find(mUSMContextInfo.DeviceAllocVector.begin(),
                    mUSMContextInfo.DeviceAllocVector.end(), ptr));
      break;
    case CL_MEM_TYPE_SHARED_INTEL:
      mUSMContextInfo.SharedAllocVector.erase(
          std::find(mUSMContextInfo.SharedAllocVector.begin(),
                    mUSMContextInfo.SharedAllocVector.end(), ptr));
      break;
    default:
      assert(0 && "unsupported!");
      break;
    }

    mUSMContextInfo.AllocMap.erase(ptr);

    clSVMFree(context, const_cast<void *>(ptr));
    ptr = nullptr;

    return CL_SUCCESS;
  }

  return CL_INVALID_MEM_OBJECT;
}

cl_int CLUSM::getMemAllocInfoINTEL(cl_context context, const void *ptr,
                                   cl_mem_info_intel param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
  std::lock_guard<std::mutex> guard(mLock);
  if (ptr == nullptr) {
    return CL_INVALID_VALUE;
  }

  if (mUSMContextInfo.AllocMap.empty()) {
    // No pointers allocated?
    return CL_INVALID_MEM_OBJECT; // TODO: new error code?
  }

  CUSMAllocMap::iterator iter = mUSMContextInfo.AllocMap.lower_bound(ptr);

  if (iter->first != ptr) {
    if (iter == mUSMContextInfo.AllocMap.begin()) {
      // This pointer is not in the map.
      return CL_INVALID_MEM_OBJECT;
    }

    // Go to the previous iterator.
    --iter;
  }

  const SUSMAllocInfo &allocInfo = iter->second;

  auto startPtr = static_cast<const char *>(allocInfo.BaseAddress);
  auto endPtr = startPtr + allocInfo.Size;
  if (ptr < startPtr || ptr >= endPtr) {
    return CL_INVALID_MEM_OBJECT;
  }

  switch (param_name) {
  case CL_MEM_ALLOC_TYPE_INTEL: {
    auto ptr =
        reinterpret_cast<cl_unified_shared_memory_type_intel *>(param_value);
    return writeParamToMemory(param_value_size, allocInfo.Type,
                              param_value_size_ret, ptr);
  }
  case CL_MEM_ALLOC_BASE_PTR_INTEL: {
    auto ptr = reinterpret_cast<const void **>(param_value);
    return writeParamToMemory(param_value_size, allocInfo.BaseAddress,
                              param_value_size_ret, ptr);
  }
  case CL_MEM_ALLOC_SIZE_INTEL: {
    auto ptr = reinterpret_cast<size_t *>(param_value);
    return writeParamToMemory(param_value_size, allocInfo.Size,
                              param_value_size_ret, ptr);
  }
  default:
    break;
  }

  return CL_INVALID_VALUE;
}

cl_int CLUSM::setKernelExecInfo(cl_kernel kernel,
                                cl_kernel_exec_info param_name,
                                size_t param_value_size,
                                const void *param_value) {
  std::lock_guard<std::mutex> guard(mLock);

  cl_int retVal = CL_INVALID_VALUE;

  switch (param_name) {
  case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
    if (param_value_size == sizeof(cl_bool)) {
      SUSMKernelInfo &kernelInfo = mUSMKernelInfoMap[kernel];
      auto pBool = reinterpret_cast<const cl_bool *>(param_value);

      kernelInfo.IndirectHostAccess = (pBool[0] == CL_TRUE);
      retVal = CL_SUCCESS;
    }
    break;
  case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
    if (param_value_size == sizeof(cl_bool)) {
      SUSMKernelInfo &kernelInfo = mUSMKernelInfoMap[kernel];
      auto pBool = reinterpret_cast<const cl_bool *>(param_value);

      kernelInfo.IndirectDeviceAccess = (pBool[0] == CL_TRUE);
      retVal = CL_SUCCESS;
    }
    break;
  case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL:
    if (param_value_size == sizeof(cl_bool)) {
      SUSMKernelInfo &kernelInfo = mUSMKernelInfoMap[kernel];
      auto pBool = reinterpret_cast<const cl_bool *>(param_value);

      kernelInfo.IndirectSharedAccess = (pBool[0] == CL_TRUE);
      retVal = CL_SUCCESS;
    }
    break;
  case CL_KERNEL_EXEC_INFO_SVM_PTRS: {
    SUSMKernelInfo &kernelInfo = mUSMKernelInfoMap[kernel];
    auto pPtrs = reinterpret_cast<void **>(const_cast<void *>(param_value));
    size_t numPtrs = param_value_size / sizeof(void *);

    kernelInfo.SVMPtrs.clear();
    kernelInfo.SVMPtrs.reserve(numPtrs);
    kernelInfo.SVMPtrs.insert(kernelInfo.SVMPtrs.begin(), pPtrs,
                              pPtrs + numPtrs);

    // Don't set CL_SUCCESS so the call passes through.
  } break;
  default:
    break;
  }

  return retVal;
}

cl_int CLUSM::setKernelIndirectUSMExecInfo(cl_command_queue commandQueue,
                                           cl_kernel kernel) {
  const SUSMKernelInfo &usmKernelInfo = mUSMKernelInfoMap[kernel];

  cl_int errorCode = CL_SUCCESS;

  if (usmKernelInfo.IndirectHostAccess || usmKernelInfo.IndirectDeviceAccess ||
      usmKernelInfo.IndirectSharedAccess) {
    // If we supported multiple contexts, we'd get the context from
    // the queue, and map it to a USM context info structure here.

    const SUSMContextInfo &usmContextInfo = mUSMContextInfo;

    // If we supported multiple devices, we'd get the device from
    // the queue and map it to the device's allocation vector here.

    std::lock_guard<std::mutex> guard(mLock);

    bool hasSVMPtrs = !usmKernelInfo.SVMPtrs.empty();
    bool setHostAllocs = !usmContextInfo.HostAllocVector.empty() &&
                         usmKernelInfo.IndirectHostAccess;
    bool setDeviceAllocs = !usmContextInfo.DeviceAllocVector.empty() &&
                           usmKernelInfo.IndirectDeviceAccess;
    bool setSharedAllocs = !usmContextInfo.SharedAllocVector.empty() &&
                           usmKernelInfo.IndirectSharedAccess;

    bool fastPath = (hasSVMPtrs == false) &&
                    ((!setHostAllocs && !setDeviceAllocs && !setSharedAllocs) ||
                     (setHostAllocs && !setDeviceAllocs && !setSharedAllocs) ||
                     (!setHostAllocs && setDeviceAllocs && !setSharedAllocs) ||
                     (!setHostAllocs && !setDeviceAllocs && setSharedAllocs));

    if (fastPath) {
      if (setHostAllocs) {
        size_t count = usmContextInfo.HostAllocVector.size();

        errorCode = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                        count * sizeof(void *),
                                        usmContextInfo.HostAllocVector.data());
      }
      if (setDeviceAllocs) {
        size_t count = usmContextInfo.DeviceAllocVector.size();

        errorCode = clSetKernelExecInfo(
            kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, count * sizeof(void *),
            usmContextInfo.DeviceAllocVector.data());
      }
      if (setSharedAllocs) {
        size_t count = usmContextInfo.SharedAllocVector.size();

        errorCode = clSetKernelExecInfo(
            kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, count * sizeof(void *),
            usmContextInfo.SharedAllocVector.data());
      }
    } else {
      size_t count = usmKernelInfo.SVMPtrs.size() + setHostAllocs
                         ? usmContextInfo.HostAllocVector.size()
                         : 0 + setDeviceAllocs
                               ? usmContextInfo.DeviceAllocVector.size()
                               : 0 + setSharedAllocs
                                     ? usmContextInfo.SharedAllocVector.size()
                                     : 0;

      std::vector<const void *> combined;
      combined.reserve(count);

      combined.insert(combined.end(), usmKernelInfo.SVMPtrs.begin(),
                      usmKernelInfo.SVMPtrs.end());
      if (setHostAllocs) {
        combined.insert(combined.end(), usmContextInfo.HostAllocVector.begin(),
                        usmContextInfo.HostAllocVector.end());
      }
      if (setDeviceAllocs) {
        combined.insert(combined.end(),
                        usmContextInfo.DeviceAllocVector.begin(),
                        usmContextInfo.DeviceAllocVector.end());
      }
      if (setSharedAllocs) {
        combined.insert(combined.end(),
                        usmContextInfo.SharedAllocVector.begin(),
                        usmContextInfo.SharedAllocVector.end());
      }

      errorCode = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                      count * sizeof(void *), combined.data());
    }
  }

  return errorCode;
}

template <class T>
cl_int CLUSM::writeParamToMemory(size_t param_value_size, T param,
                                 size_t *param_value_size_ret,
                                 T *pointer) const {
  cl_int errorCode = CL_SUCCESS;

  if (pointer != nullptr) {
    if (param_value_size < sizeof(param)) {
      errorCode = CL_INVALID_VALUE;
    } else {
      *pointer = param;
    }
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = sizeof(param);
  }

  return errorCode;
}

} // namespace usm
} // namespace detail
} // namespace sycl
} // namespace cl

//==---------- pi_opencl.cpp - OpenCL Plugin -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \defgroup sycl_pi_ocl OpenCL Plugin
/// \ingroup sycl_pi

/// \file pi_opencl.cpp
/// Implementation of OpenCL Plugin. It is the interface between device-agnostic
/// SYCL runtime layer and underlying OpenCL runtime.
///
/// \ingroup sycl_pi_ocl

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <pi_opencl.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/pi.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define CHECK_ERR_SET_NULL_RET(err, ptr, reterr)                               \
  if (err != CL_SUCCESS) {                                                     \
    if (ptr != nullptr)                                                        \
      *ptr = nullptr;                                                          \
    return cast<pi_result>(reterr);                                            \
  }

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value) {
  // TODO: see if more sanity checks are possible.
  static_assert(sizeof(From) == sizeof(To), "cast failed size check");
  return (To)(value);
}

// Older versions of GCC don't like "const" here
#if defined(__GNUC__) && (__GNUC__ < 7 || (__GNU__C == 7 && __GNUC_MINOR__ < 2))
#define CONSTFIX constexpr
#else
#define CONSTFIX const
#endif

// Names of USM functions that are queried from OpenCL
CONSTFIX char clHostMemAllocName[] = "clHostMemAllocINTEL";
CONSTFIX char clDeviceMemAllocName[] = "clDeviceMemAllocINTEL";
CONSTFIX char clSharedMemAllocName[] = "clSharedMemAllocINTEL";
CONSTFIX char clMemBlockingFreeName[] = "clMemBlockingFreeINTEL";
CONSTFIX char clCreateBufferWithPropertiesName[] =
    "clCreateBufferWithPropertiesINTEL";
CONSTFIX char clSetKernelArgMemPointerName[] = "clSetKernelArgMemPointerINTEL";
CONSTFIX char clEnqueueMemsetName[] = "clEnqueueMemsetINTEL";
CONSTFIX char clEnqueueMemcpyName[] = "clEnqueueMemcpyINTEL";
CONSTFIX char clGetMemAllocInfoName[] = "clGetMemAllocInfoINTEL";
CONSTFIX char clSetProgramSpecializationConstantName[] =
    "clSetProgramSpecializationConstant";
CONSTFIX char clGetDeviceFunctionPointerName[] =
    "clGetDeviceFunctionPointerINTEL";
CONSTFIX char clEnqueueWriteGlobalVariableName[] =
    "clEnqueueWriteGlobalVariableINTEL";
CONSTFIX char clEnqueueReadGlobalVariableName[] =
    "clEnqueueReadGlobalVariableINTEL";
// Names of host pipe functions queried from OpenCL
CONSTFIX char clEnqueueReadHostPipeName[] = "clEnqueueReadHostPipeINTEL";
CONSTFIX char clEnqueueWriteHostPipeName[] = "clEnqueueWriteHostPipeINTEL";

#undef CONSTFIX

// Global variables for PI_ERROR_PLUGIN_SPECIFIC_ERROR
constexpr size_t MaxMessageSize = 256;
thread_local pi_result ErrorMessageCode = PI_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] static void setErrorMessage(const char *message,
                                             pi_result error_code) {
  assert(strlen(message) <= MaxMessageSize);
  strcpy(ErrorMessage, message);
  ErrorMessageCode = error_code;
}

// Returns plugin specific error and warning messages
pi_result piPluginGetLastError(char **message) {
  *message = &ErrorMessage[0];
  return ErrorMessageCode;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-cl-opt-disable' for frontend_option = -O0 and '' for others.
pi_result piPluginGetBackendOption(pi_platform, const char *frontend_option,
                                   const char **backend_option) {
  using namespace std::literals;
  if (frontend_option == nullptr)
    return PI_ERROR_INVALID_VALUE;
  if (frontend_option == ""sv) {
    *backend_option = "";
    return PI_SUCCESS;
  }
  if (!strcmp(frontend_option, "-O0")) {
    *backend_option = "-cl-opt-disable";
    return PI_SUCCESS;
  }
  if (frontend_option == "-O1"sv || frontend_option == "-O2"sv ||
      frontend_option == "-O3"sv) {
    *backend_option = "";
    return PI_SUCCESS;
  }
  return PI_ERROR_INVALID_VALUE;
}

static cl_int getPlatformVersion(cl_platform_id plat,
                                 OCLV::OpenCLVersion &version) {
  cl_int ret_err = CL_INVALID_VALUE;

  size_t platVerSize = 0;
  ret_err =
      clGetPlatformInfo(plat, CL_PLATFORM_VERSION, 0, nullptr, &platVerSize);

  std::string platVer(platVerSize, '\0');
  ret_err = clGetPlatformInfo(plat, CL_PLATFORM_VERSION, platVerSize,
                              platVer.data(), nullptr);

  if (ret_err != CL_SUCCESS)
    return ret_err;

  version = OCLV::OpenCLVersion(platVer);
  if (!version.isValid())
    return CL_INVALID_PLATFORM;

  return ret_err;
}

static cl_int getDeviceVersion(cl_device_id dev, OCLV::OpenCLVersion &version) {
  cl_int ret_err = CL_INVALID_VALUE;

  size_t devVerSize = 0;
  ret_err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, nullptr, &devVerSize);

  std::string devVer(devVerSize, '\0');
  ret_err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, devVerSize, devVer.data(),
                            nullptr);

  if (ret_err != CL_SUCCESS)
    return ret_err;

  version = OCLV::OpenCLVersion(devVer);
  if (!version.isValid())
    return CL_INVALID_DEVICE;

  return ret_err;
}

static cl_int checkDeviceExtensions(cl_device_id dev,
                                    const std::vector<std::string> &exts,
                                    bool &supported) {
  cl_int ret_err = CL_INVALID_VALUE;

  size_t extSize = 0;
  ret_err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &extSize);

  std::string extStr(extSize, '\0');
  ret_err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, extSize, extStr.data(),
                            nullptr);

  if (ret_err != CL_SUCCESS)
    return ret_err;

  supported = true;
  for (const std::string &ext : exts)
    if (!(supported = (extStr.find(ext) != std::string::npos)))
      break;

  return ret_err;
}

using clGetDeviceFunctionPointer_fn = CL_API_ENTRY
cl_int(CL_API_CALL *)(cl_device_id device, cl_program program,
                      const char *FuncName, cl_ulong *ret_ptr);

using clEnqueueWriteGlobalVariable_fn = CL_API_ENTRY
cl_int(CL_API_CALL *)(cl_command_queue, cl_program, const char *, cl_bool,
                      size_t, size_t, const void *, cl_uint, const cl_event *,
                      cl_event *);

using clEnqueueReadGlobalVariable_fn = CL_API_ENTRY
cl_int(CL_API_CALL *)(cl_command_queue, cl_program, const char *, cl_bool,
                      size_t, size_t, void *, cl_uint, const cl_event *,
                      cl_event *);

using clSetProgramSpecializationConstant_fn = CL_API_ENTRY
cl_int(CL_API_CALL *)(cl_program program, cl_uint spec_id, size_t spec_size,
                      const void *spec_value);

template <typename T> struct FuncPtrCache {
  std::map<cl_context, T> Map;
  std::mutex Mutex;
};

// FIXME: There's currently no mechanism for cleaning up this cache, meaning
// that it is invalidated whenever a context is destroyed. This could lead to
// reusing an invalid function pointer if another context happends to have the
// same native handle.
struct ExtFuncPtrCacheT {
  FuncPtrCache<clHostMemAllocINTEL_fn> clHostMemAllocINTELCache;
  FuncPtrCache<clDeviceMemAllocINTEL_fn> clDeviceMemAllocINTELCache;
  FuncPtrCache<clSharedMemAllocINTEL_fn> clSharedMemAllocINTELCache;
  FuncPtrCache<clGetDeviceFunctionPointer_fn> clGetDeviceFunctionPointerCache;
  FuncPtrCache<clCreateBufferWithPropertiesINTEL_fn>
      clCreateBufferWithPropertiesINTELCache;
  FuncPtrCache<clMemBlockingFreeINTEL_fn> clMemBlockingFreeINTELCache;
  FuncPtrCache<clSetKernelArgMemPointerINTEL_fn>
      clSetKernelArgMemPointerINTELCache;
  FuncPtrCache<clEnqueueMemsetINTEL_fn> clEnqueueMemsetINTELCache;
  FuncPtrCache<clEnqueueMemcpyINTEL_fn> clEnqueueMemcpyINTELCache;
  FuncPtrCache<clGetMemAllocInfoINTEL_fn> clGetMemAllocInfoINTELCache;
  FuncPtrCache<clEnqueueWriteGlobalVariable_fn>
      clEnqueueWriteGlobalVariableCache;
  FuncPtrCache<clEnqueueReadGlobalVariable_fn> clEnqueueReadGlobalVariableCache;
  FuncPtrCache<clEnqueueReadHostPipeINTEL_fn> clEnqueueReadHostPipeINTELCache;
  FuncPtrCache<clEnqueueWriteHostPipeINTEL_fn> clEnqueueWriteHostPipeINTELCache;
  FuncPtrCache<clSetProgramSpecializationConstant_fn>
      clSetProgramSpecializationConstantCache;
};
// A raw pointer is used here since the lifetime of this map has to be tied to
// piTeardown to avoid issues with static destruction order (a user application
// might have static objects that indirectly access this cache in their
// destructor).
static ExtFuncPtrCacheT *ExtFuncPtrCache = new ExtFuncPtrCacheT();

// USM helper function to get an extension function pointer
template <typename T>
static pi_result getExtFuncFromContext(cl_context context,
                                       FuncPtrCache<T> &FPtrCache,
                                       const char *FuncName, T *fptr) {
  // TODO
  // Potentially redo caching as PI interface changes.
  // if cached, return cached FuncPtr
  std::lock_guard<std::mutex> CacheLock{FPtrCache.Mutex};
  std::map<cl_context, T> &FPtrMap = FPtrCache.Map;
  auto It = FPtrMap.find(context);
  if (It != FPtrMap.end()) {
    auto F = It->second;
    // if cached that extension is not available return nullptr and
    // PI_ERROR_INVALID_VALUE
    *fptr = F;
    return F ? PI_SUCCESS : PI_ERROR_INVALID_VALUE;
  }

  cl_uint deviceCount;
  cl_int ret_err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
                                    sizeof(cl_uint), &deviceCount, nullptr);

  if (ret_err != CL_SUCCESS || deviceCount < 1) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  std::vector<cl_device_id> devicesInCtx(deviceCount);
  ret_err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             deviceCount * sizeof(cl_device_id),
                             devicesInCtx.data(), nullptr);

  if (ret_err != CL_SUCCESS) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  cl_platform_id curPlatform;
  ret_err = clGetDeviceInfo(devicesInCtx[0], CL_DEVICE_PLATFORM,
                            sizeof(cl_platform_id), &curPlatform, nullptr);

  if (ret_err != CL_SUCCESS) {
    return PI_ERROR_INVALID_CONTEXT;
  }

  T FuncPtr =
      (T)clGetExtensionFunctionAddressForPlatform(curPlatform, FuncName);

  if (!FuncPtr) {
    // Cache that the extension is not available
    FPtrMap[context] = nullptr;
    return PI_ERROR_INVALID_VALUE;
  }

  *fptr = FuncPtr;
  FPtrMap[context] = FuncPtr;

  return cast<pi_result>(ret_err);
}

/// Enables indirect access of pointers in kernels.
/// Necessary to avoid telling CL about every pointer that might be used.
///
/// \param kernel is the kernel to be launched
static pi_result USMSetIndirectAccess(pi_kernel kernel) {
  // We test that each alloc type is supported before we actually try to
  // set KernelExecInfo.
  cl_bool TrueVal = CL_TRUE;
  clHostMemAllocINTEL_fn HFunc = nullptr;
  clSharedMemAllocINTEL_fn SFunc = nullptr;
  clDeviceMemAllocINTEL_fn DFunc = nullptr;
  cl_context CLContext;
  cl_int CLErr = clGetKernelInfo(cast<cl_kernel>(kernel), CL_KERNEL_CONTEXT,
                                 sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  getExtFuncFromContext<clHostMemAllocINTEL_fn>(
      CLContext, ExtFuncPtrCache->clHostMemAllocINTELCache, clHostMemAllocName,
      &HFunc);
  if (HFunc) {
    clSetKernelExecInfo(cast<cl_kernel>(kernel),
                        CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                        sizeof(cl_bool), &TrueVal);
  }

  getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
      CLContext, ExtFuncPtrCache->clDeviceMemAllocINTELCache,
      clDeviceMemAllocName, &DFunc);
  if (DFunc) {
    clSetKernelExecInfo(cast<cl_kernel>(kernel),
                        CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                        sizeof(cl_bool), &TrueVal);
  }

  getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
      CLContext, ExtFuncPtrCache->clSharedMemAllocINTELCache,
      clSharedMemAllocName, &SFunc);
  if (SFunc) {
    clSetKernelExecInfo(cast<cl_kernel>(kernel),
                        CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                        sizeof(cl_bool), &TrueVal);
  }
  return PI_SUCCESS;
}

extern "C" {

pi_result piDeviceGetInfo(pi_device device, pi_device_info paramName,
                          size_t paramValueSize, void *paramValue,
                          size_t *paramValueSizeRet) {
  switch (paramName) {
    // TODO: Check regularly to see if support in enabled in OpenCL.
    // Intel GPU EU device-specific information extensions.
    // Some of the queries are enabled by cl_intel_device_attribute_query
    // extension, but it's not yet in the Registry.
  case PI_DEVICE_INFO_PCI_ADDRESS:
  case PI_DEVICE_INFO_GPU_EU_COUNT:
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case PI_DEVICE_INFO_GPU_SLICES:
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    // TODO: Check if device UUID extension is enabled in OpenCL.
    // For details about Intel UUID extension, see
    // sycl/doc/extensions/supported/sycl_ext_intel_device_info.md
  case PI_DEVICE_INFO_UUID:
    return PI_ERROR_INVALID_VALUE;
  case PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    // This query is missing before OpenCL 3.0
    // Check version and handle appropriately
    OCLV::OpenCLVersion devVer;
    cl_device_id deviceID = cast<cl_device_id>(device);
    cl_int ret_err = getDeviceVersion(deviceID, devVer);
    if (ret_err != CL_SUCCESS) {
      return cast<pi_result>(ret_err);
    }

    // Minimum required capability to be returned
    // For OpenCL 1.2, this is all that is required
    pi_memory_order_capabilities capabilities = PI_MEMORY_ORDER_RELAXED;

    if (devVer >= OCLV::V3_0) {
      // For OpenCL >=3.0, the query should be implemented
      cl_device_atomic_capabilities cl_capabilities = 0;
      cl_int ret_err = clGetDeviceInfo(
          deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
          sizeof(cl_device_atomic_capabilities), &cl_capabilities, nullptr);
      if (ret_err != CL_SUCCESS)
        return cast<pi_result>(ret_err);

      // Mask operation to only consider atomic_memory_order* capabilities
      cl_int mask = CL_DEVICE_ATOMIC_ORDER_RELAXED |
                    CL_DEVICE_ATOMIC_ORDER_ACQ_REL |
                    CL_DEVICE_ATOMIC_ORDER_SEQ_CST;
      cl_capabilities &= mask;

      // The memory order capabilities are hierarchical, if one is implied, all
      // preceding capbilities are implied as well. Especially in the case of
      // ACQ_REL.
      if (cl_capabilities & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) {
        capabilities |= PI_MEMORY_ORDER_SEQ_CST;
      }
      if (cl_capabilities & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) {
        capabilities |= PI_MEMORY_ORDER_ACQ_REL | PI_MEMORY_ORDER_ACQUIRE |
                        PI_MEMORY_ORDER_RELEASE;
      }
    } else if (devVer >= OCLV::V2_0) {
      // For OpenCL 2.x, return all capabilities
      // (https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_memory_consistency_model)
      capabilities |= PI_MEMORY_ORDER_ACQUIRE | PI_MEMORY_ORDER_RELEASE |
                      PI_MEMORY_ORDER_ACQ_REL | PI_MEMORY_ORDER_SEQ_CST;
    }

    if (paramValue) {
      if (paramValueSize < sizeof(pi_memory_order_capabilities))
        return static_cast<pi_result>(CL_INVALID_VALUE);

      std::memcpy(paramValue, &capabilities, sizeof(capabilities));
    }

    if (paramValueSizeRet)
      *paramValueSizeRet = sizeof(capabilities);

    return static_cast<pi_result>(CL_SUCCESS);
  }
  case PI_EXT_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    // Initialize result to minimum mandated capabilities according to
    // SYCL2020 4.6.3.2
    // Because scopes are hierarchical, wider scopes support all narrower
    // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
    // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
    pi_memory_scope_capabilities result = PI_MEMORY_SCOPE_WORK_ITEM |
                                          PI_MEMORY_SCOPE_SUB_GROUP |
                                          PI_MEMORY_SCOPE_WORK_GROUP;

    OCLV::OpenCLVersion devVer;

    cl_device_id deviceID = cast<cl_device_id>(device);
    cl_int ret_err = getDeviceVersion(deviceID, devVer);
    if (ret_err != CL_SUCCESS)
      return static_cast<pi_result>(ret_err);

    cl_device_atomic_capabilities devCapabilities = 0;
    if (devVer >= OCLV::V3_0) {
      ret_err = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                                sizeof(cl_device_atomic_capabilities),
                                &devCapabilities, nullptr);
      if (ret_err != CL_SUCCESS)
        return static_cast<pi_result>(ret_err);
      assert((devCapabilities & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) &&
             "Violates minimum mandated guarantee");

      // Because scopes are hierarchical, wider scopes support all narrower
      // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
      // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
      // We already initialized to these minimum mandated capabilities. Just
      // check wider scopes.
      if (devCapabilities & CL_DEVICE_ATOMIC_SCOPE_DEVICE) {
        result |= PI_MEMORY_SCOPE_DEVICE;
      }

      if (devCapabilities & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) {
        result |= PI_MEMORY_SCOPE_SYSTEM;
      }

    } else {
      // This info is only available in OpenCL version >= 3.0
      // Just return minimum mandated capabilities for older versions.
      // OpenCL 1.x minimum mandated capabilities are WORK_GROUP, we
      // already initialized using it.
      if (devVer >= OCLV::V2_0) {
        // OpenCL 2.x minimum mandated capabilities are WORK_GROUP | DEVICE |
        // ALL_DEVICES
        result |= PI_MEMORY_SCOPE_DEVICE | PI_MEMORY_SCOPE_SYSTEM;
      }
    }
    if (paramValue) {
      if (paramValueSize < sizeof(cl_device_atomic_capabilities))
        return PI_ERROR_INVALID_VALUE;

      std::memcpy(paramValue, &result, sizeof(result));
    }
    if (paramValueSizeRet)
      *paramValueSizeRet = sizeof(result);
    return PI_SUCCESS;
  }
  case PI_EXT_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES: {
    // Initialize result to minimum mandated capabilities according to
    // SYCL2020 4.6.3.2
    pi_memory_order_capabilities result =
        PI_MEMORY_ORDER_RELAXED | PI_MEMORY_ORDER_ACQUIRE |
        PI_MEMORY_ORDER_RELEASE | PI_MEMORY_ORDER_ACQ_REL;

    OCLV::OpenCLVersion devVer;

    cl_device_id deviceID = cast<cl_device_id>(device);
    cl_int ret_err = getDeviceVersion(deviceID, devVer);
    if (ret_err != CL_SUCCESS)
      return static_cast<pi_result>(ret_err);

    cl_device_atomic_capabilities devCapabilities = 0;
    if (devVer >= OCLV::V3_0) {
      ret_err = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                                sizeof(cl_device_atomic_capabilities),
                                &devCapabilities, nullptr);
      if (ret_err != CL_SUCCESS)
        return static_cast<pi_result>(ret_err);
      assert((devCapabilities & CL_DEVICE_ATOMIC_ORDER_RELAXED) &&
             "Violates minimum mandated guarantee");
      assert((devCapabilities & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) &&
             "Violates minimum mandated guarantee");

      // We already initialized to minimum mandated capabilities. Just
      // check stronger orders.
      if (devCapabilities & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) {
        result |= PI_MEMORY_ORDER_SEQ_CST;
      }

    } else {
      // This info is only available in OpenCL version >= 3.0
      // Just return minimum mandated capabilities for older versions.
      // OpenCL 1.x minimum mandated capabilities are RELAXED | ACQ_REL, we
      // already initialized using these.
      if (devVer >= OCLV::V2_0) {
        // OpenCL 2.x minimum mandated capabilities are RELAXED | ACQ_REL |
        // SEQ_CST
        result |= PI_MEMORY_ORDER_SEQ_CST;
      }
    }
    if (paramValue) {
      if (paramValueSize < sizeof(cl_device_atomic_capabilities))
        return PI_ERROR_INVALID_VALUE;

      std::memcpy(paramValue, &result, sizeof(result));
    }
    if (paramValueSizeRet)
      *paramValueSizeRet = sizeof(result);
    return PI_SUCCESS;
  }
  case PI_EXT_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // Initialize result to minimum mandated capabilities according to
    // SYCL2020 4.6.3.2.
    // Because scopes are hierarchical, wider scopes support all narrower
    // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
    // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
    pi_memory_scope_capabilities result = PI_MEMORY_SCOPE_WORK_ITEM |
                                          PI_MEMORY_SCOPE_SUB_GROUP |
                                          PI_MEMORY_SCOPE_WORK_GROUP;

    OCLV::OpenCLVersion devVer;

    cl_device_id deviceID = cast<cl_device_id>(device);
    cl_int ret_err = getDeviceVersion(deviceID, devVer);
    if (ret_err != CL_SUCCESS)
      return static_cast<pi_result>(ret_err);

    cl_device_atomic_capabilities devCapabilities = 0;
    if (devVer >= OCLV::V3_0) {
      ret_err = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                                sizeof(cl_device_atomic_capabilities),
                                &devCapabilities, nullptr);
      if (ret_err != CL_SUCCESS)
        return static_cast<pi_result>(ret_err);
      assert((devCapabilities & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) &&
             "Violates minimum mandated guarantee");

      // Because scopes are hierarchical, wider scopes support all narrower
      // scopes. At a minimum, each device must support WORK_ITEM, SUB_GROUP and
      // WORK_GROUP. (https://github.com/KhronosGroup/SYCL-Docs/pull/382)
      // We already initialized to these minimum mandated capabilities. Just
      // check wider scopes.
      if (devCapabilities & CL_DEVICE_ATOMIC_SCOPE_DEVICE) {
        result |= PI_MEMORY_SCOPE_DEVICE;
      }

      if (devCapabilities & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) {
        result |= PI_MEMORY_SCOPE_SYSTEM;
      }

    } else {
      // This info is only available in OpenCL version >= 3.0
      // Just return minimum mandated capabilities for older versions.
      // OpenCL 1.x minimum mandated capabilities are WORK_GROUP, we
      // already initialized using it.
      if (devVer >= OCLV::V2_0) {
        // OpenCL 2.x minimum mandated capabilities are WORK_GROUP | DEVICE |
        // ALL_DEVICES
        result |= PI_MEMORY_SCOPE_DEVICE | PI_MEMORY_SCOPE_SYSTEM;
      }
    }
    if (paramValue) {
      if (paramValueSize < sizeof(cl_device_atomic_capabilities))
        return PI_ERROR_INVALID_VALUE;

      std::memcpy(paramValue, &result, sizeof(result));
    }
    if (paramValueSizeRet)
      *paramValueSizeRet = sizeof(result);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_ATOMIC_64: {
    cl_int ret_err = CL_SUCCESS;
    cl_bool result = CL_FALSE;
    bool supported = false;

    ret_err = checkDeviceExtensions(
        cast<cl_device_id>(device),
        {"cl_khr_int64_base_atomics", "cl_khr_int64_extended_atomics"},
        supported);
    if (ret_err != CL_SUCCESS)
      return static_cast<pi_result>(ret_err);

    result = supported;
    std::memcpy(paramValue, &result, sizeof(cl_bool));
    return PI_SUCCESS;
  }
  case PI_EXT_ONEAPI_DEVICE_INFO_BFLOAT16_MATH_FUNCTIONS: {
    // bfloat16 math functions are not yet supported on Intel GPUs.
    cl_bool result = false;
    std::memcpy(paramValue, &result, sizeof(cl_bool));
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_IMAGE_SRGB: {
    cl_bool result = true;
    std::memcpy(paramValue, &result, sizeof(cl_bool));
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_BUILD_ON_SUBDEVICE: {
    cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
    cl_int res = clGetDeviceInfo(cast<cl_device_id>(device), CL_DEVICE_TYPE,
                                 sizeof(cl_device_type), &devType, nullptr);

    // FIXME: here we assume that program built for a root GPU device can be
    // used on its sub-devices without re-building
    cl_bool result = (res == CL_SUCCESS) && (devType == CL_DEVICE_TYPE_GPU);
    std::memcpy(paramValue, &result, sizeof(cl_bool));
    return PI_SUCCESS;
  }
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D:
    // Returns the maximum sizes of a work group for each dimension one
    // could use to submit a kernel. There is no such query defined in OpenCL
    // so we'll return the maximum value.
    {
      if (paramValueSizeRet)
        *paramValueSizeRet = paramValueSize;
      static constexpr size_t Max = (std::numeric_limits<size_t>::max)();
      size_t *out = cast<size_t *>(paramValue);
      if (paramValueSize >= sizeof(size_t))
        out[0] = Max;
      if (paramValueSize >= 2 * sizeof(size_t))
        out[1] = Max;
      if (paramValueSize >= 3 * sizeof(size_t))
        out[2] = Max;
      return PI_SUCCESS;
    }
  case PI_EXT_INTEL_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    pi_int32 result = 1;
    std::memcpy(paramValue, &result, sizeof(pi_int32));
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Corresponding OpenCL query is only available starting with OpenCL 2.1 and
    // we have to emulate it on older OpenCL runtimes.
    OCLV::OpenCLVersion version;
    cl_int err = getDeviceVersion(cast<cl_device_id>(device), version);
    if (err != CL_SUCCESS)
      return static_cast<pi_result>(err);

    if (version >= OCLV::V2_1) {
      err = clGetDeviceInfo(cast<cl_device_id>(device),
                            cast<cl_device_info>(paramName), paramValueSize,
                            paramValue, paramValueSizeRet);
      if (err != CL_SUCCESS)
        return static_cast<pi_result>(err);

      if (paramValue && *static_cast<cl_uint *>(paramValue) == 0u) {
        // OpenCL returns 0 if sub-groups are not supported, but SYCL 2020 spec
        // says that minimum possible value is 1.
        cl_uint value = 1u;
        std::memcpy(paramValue, &value, sizeof(cl_uint));
      }

      return static_cast<pi_result>(err);
    }

    // Otherwise, we can't query anything, because even cl_khr_subgroups does
    // not provide similar query. Therefore, simply return minimum possible
    // value 1 here.
    if (paramValue && paramValueSize < sizeof(cl_uint))
      return static_cast<pi_result>(CL_INVALID_VALUE);
    if (paramValueSizeRet)
      *paramValueSizeRet = sizeof(cl_uint);

    if (paramValue) {
      cl_uint value = 1u;
      std::memcpy(paramValue, &value, sizeof(cl_uint));
    }

    return static_cast<pi_result>(CL_SUCCESS);
  }
  case PI_DEVICE_INFO_BACKEND_VERSION: {
    // TODO: return some meaningful for backend_version below
    const char *value = "";
    size_t valueSize = (strlen(value) + 1) * sizeof(char);
    if (paramValue)
      std::memcpy(paramValue, value, valueSize);
    if (paramValueSizeRet != nullptr)
      *paramValueSizeRet = valueSize;
    return PI_SUCCESS;
  }
  case PI_EXT_INTEL_DEVICE_INFO_MEM_CHANNEL_SUPPORT: {
    cl_int ret_err = CL_SUCCESS;
    cl_bool result = CL_FALSE;
    bool supported = false;

    ret_err =
        checkDeviceExtensions(cast<cl_device_id>(device),
                              {"cl_intel_mem_channel_property"}, supported);
    if (ret_err != CL_SUCCESS)
      return static_cast<pi_result>(ret_err);

    result = supported;
    std::memcpy(paramValue, &result, sizeof(cl_bool));
    return PI_SUCCESS;
  }
  default:
    cl_int result = clGetDeviceInfo(
        cast<cl_device_id>(device), cast<cl_device_info>(paramName),
        paramValueSize, paramValue, paramValueSizeRet);
    return static_cast<pi_result>(result);
  }
}

pi_result piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                         pi_uint32 *num_platforms) {
  cl_int result = clGetPlatformIDs(cast<cl_uint>(num_entries),
                                   cast<cl_platform_id *>(platforms),
                                   cast<cl_uint *>(num_platforms));

  // Absorb the CL_PLATFORM_NOT_FOUND_KHR and just return 0 in num_platforms
  if (result == CL_PLATFORM_NOT_FOUND_KHR) {
    assert(num_platforms != 0);
    *num_platforms = 0;
    result = PI_SUCCESS;
  }
  return static_cast<pi_result>(result);
}

pi_result piPlatformGetInfo(pi_platform platform, pi_platform_info paramName,
                            size_t paramValueSize, void *paramValue,
                            size_t *paramValueSizeRet) {

  switch (paramName) {
  case PI_EXT_PLATFORM_INFO_BACKEND: {
    pi_platform_backend result = PI_EXT_PLATFORM_BACKEND_OPENCL;
    if (paramValue) {
      if (paramValueSize < sizeof(result))
        return PI_ERROR_INVALID_VALUE;
      std::memcpy(paramValue, &result, sizeof(result));
    }
    if (paramValueSizeRet)
      *paramValueSizeRet = sizeof(result);
    return PI_SUCCESS;
  }
  default: {
    cl_int result = clGetPlatformInfo(
        cast<cl_platform_id>(platform), cast<cl_platform_info>(paramName),
        paramValueSize, paramValue, paramValueSizeRet);
    return static_cast<pi_result>(result);
  }
  }
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle nativeHandle,
                                              pi_platform *platform) {
  assert(platform);
  assert(nativeHandle);
  *platform = reinterpret_cast<pi_platform>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piDevicesGet(pi_platform platform, pi_device_type device_type,
                       pi_uint32 num_entries, pi_device *devices,
                       pi_uint32 *num_devices) {
  cl_int result = clGetDeviceIDs(
      cast<cl_platform_id>(platform), cast<cl_device_type>(device_type),
      cast<cl_uint>(num_entries), cast<cl_device_id *>(devices),
      cast<cl_uint *>(num_devices));

  // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
  if (result == CL_DEVICE_NOT_FOUND) {
    assert(num_devices != 0);
    *num_devices = 0;
    result = PI_SUCCESS;
  }
  return cast<pi_result>(result);
}

pi_result piextDeviceSelectBinary(pi_device device, pi_device_binary *images,
                                  pi_uint32 num_images,
                                  pi_uint32 *selected_image_ind) {

  // TODO: this is a bare-bones implementation for choosing a device image
  // that would be compatible with the targeted device. An AOT-compiled
  // image is preferred over SPIR-V for known devices (i.e. Intel devices)
  // The implementation makes no effort to differentiate between multiple images
  // for the given device, and simply picks the first one compatible
  // Real implementation will use the same mechanism OpenCL ICD dispatcher
  // uses. Something like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_ERROR_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  // Choose the binary target for the provided device
  const char *image_target = nullptr;
  // Get the type of the device
  cl_device_type device_type;
  constexpr pi_uint32 invalid_ind = std::numeric_limits<pi_uint32>::max();
  cl_int ret_err =
      clGetDeviceInfo(cast<cl_device_id>(device), CL_DEVICE_TYPE,
                      sizeof(cl_device_type), &device_type, nullptr);
  if (ret_err != CL_SUCCESS) {
    *selected_image_ind = invalid_ind;
    return cast<pi_result>(ret_err);
  }

  switch (device_type) {
    // TODO: Factor out vendor specifics into a separate source
    // E.g. sycl/source/detail/vendor/intel/detail/pi_opencl.cpp?

    // We'll attempt to find an image that was AOT-compiled
    // from a SPIR-V image into an image specific for:

  case CL_DEVICE_TYPE_CPU: // OpenCL 64-bit CPU
    image_target = __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64;
    break;
  case CL_DEVICE_TYPE_GPU: // OpenCL 64-bit GEN GPU
    image_target = __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN;
    break;
  case CL_DEVICE_TYPE_ACCELERATOR: // OpenCL 64-bit FPGA
    image_target = __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA;
    break;
  default:
    // Otherwise, we'll attempt to find and JIT-compile
    // a device-independent SPIR-V image
    image_target = __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64;
    break;
  }

  // Find the appropriate device image, fallback to spirv if not found
  pi_uint32 fallback = invalid_ind;
  for (pi_uint32 i = 0; i < num_images; ++i) {
    if (strcmp(images[i]->DeviceTargetSpec, image_target) == 0) {
      *selected_image_ind = i;
      return PI_SUCCESS;
    }
    if (strcmp(images[i]->DeviceTargetSpec,
               __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0)
      fallback = i;
  }
  // Points to a spirv image, if such indeed was found
  if ((*selected_image_ind = fallback) != invalid_ind)
    return PI_SUCCESS;
  // No image can be loaded for the given device
  return PI_ERROR_INVALID_BINARY;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle nativeHandle,
                                            pi_platform, pi_device *piDevice) {
  assert(piDevice != nullptr);
  *piDevice = reinterpret_cast<pi_device>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piextQueueCreate(pi_context Context, pi_device Device,
                           pi_queue_properties *Properties, pi_queue *Queue) {
  assert(Properties);
  // Expect flags mask to be passed first.
  assert(Properties[0] == PI_QUEUE_FLAGS);
  if (Properties[0] != PI_QUEUE_FLAGS)
    return PI_ERROR_INVALID_VALUE;
  pi_queue_properties Flags = Properties[1];
  // Extra data isn't supported yet.
  assert(Properties[2] == 0);
  if (Properties[2] != 0)
    return PI_ERROR_INVALID_VALUE;
  return piQueueCreate(Context, Device, Flags, Queue);
}
pi_result piQueueCreate(pi_context context, pi_device device,
                        pi_queue_properties properties, pi_queue *queue) {
  assert(queue && "piQueueCreate failed, queue argument is null");

  cl_platform_id curPlatform;
  cl_int ret_err =
      clGetDeviceInfo(cast<cl_device_id>(device), CL_DEVICE_PLATFORM,
                      sizeof(cl_platform_id), &curPlatform, nullptr);

  CHECK_ERR_SET_NULL_RET(ret_err, queue, ret_err);

  // Check that unexpected bits are not set.
  assert(!(properties &
           ~(PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
             PI_QUEUE_FLAG_PROFILING_ENABLE | PI_QUEUE_FLAG_ON_DEVICE |
             PI_QUEUE_FLAG_ON_DEVICE_DEFAULT |
             PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS)));

  // Properties supported by OpenCL backend.
  cl_command_queue_properties SupportByOpenCL =
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE |
      CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;

  OCLV::OpenCLVersion version;
  ret_err = getPlatformVersion(curPlatform, version);

  CHECK_ERR_SET_NULL_RET(ret_err, queue, ret_err);

  if (version >= OCLV::V2_0) {
    *queue = cast<pi_queue>(clCreateCommandQueue(
        cast<cl_context>(context), cast<cl_device_id>(device),
        cast<cl_command_queue_properties>(properties) & SupportByOpenCL,
        &ret_err));
    return cast<pi_result>(ret_err);
  }

  cl_queue_properties CreationFlagProperties[] = {
      CL_QUEUE_PROPERTIES,
      cast<cl_command_queue_properties>(properties) & SupportByOpenCL, 0};
  *queue = cast<pi_queue>(clCreateCommandQueueWithProperties(
      cast<cl_context>(context), cast<cl_device_id>(device),
      CreationFlagProperties, &ret_err));
  return cast<pi_result>(ret_err);
}

pi_result piQueueGetInfo(pi_queue queue, pi_queue_info param_name,
                         size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret) {
  if (queue == nullptr) {
    return PI_ERROR_INVALID_QUEUE;
  }

  switch (param_name) {
  case PI_EXT_ONEAPI_QUEUE_INFO_EMPTY:
    // OpenCL doesn't provide API to check the status of the queue.
    return PI_ERROR_INVALID_VALUE;
  default:
    cl_int CLErr = clGetCommandQueueInfo(
        cast<cl_command_queue>(queue), cast<cl_command_queue_info>(param_name),
        param_value_size, param_value, param_value_size_ret);
    if (CLErr != CL_SUCCESS) {
      return cast<pi_result>(CLErr);
    }
  }
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle nativeHandle,
                                           int32_t NativeHandleDesc, pi_context,
                                           pi_device, bool ownNativeHandle,
                                           pi_queue_properties *Properties,
                                           pi_queue *piQueue) {
  (void)NativeHandleDesc;
  (void)ownNativeHandle;
  (void)Properties;
  assert(piQueue != nullptr);
  *piQueue = reinterpret_cast<pi_queue>(nativeHandle);
  clRetainCommandQueue(cast<cl_command_queue>(nativeHandle));
  return PI_SUCCESS;
}

pi_result piProgramCreate(pi_context context, const void *il, size_t length,
                          pi_program *res_program) {
  cl_uint deviceCount;
  cl_int ret_err =
      clGetContextInfo(cast<cl_context>(context), CL_CONTEXT_NUM_DEVICES,
                       sizeof(cl_uint), &deviceCount, nullptr);

  std::vector<cl_device_id> devicesInCtx(deviceCount);

  if (ret_err != CL_SUCCESS || deviceCount < 1) {
    if (res_program != nullptr)
      *res_program = nullptr;
    return cast<pi_result>(CL_INVALID_CONTEXT);
  }

  ret_err = clGetContextInfo(cast<cl_context>(context), CL_CONTEXT_DEVICES,
                             deviceCount * sizeof(cl_device_id),
                             devicesInCtx.data(), nullptr);

  CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

  cl_platform_id curPlatform;
  ret_err = clGetDeviceInfo(devicesInCtx[0], CL_DEVICE_PLATFORM,
                            sizeof(cl_platform_id), &curPlatform, nullptr);

  CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

  OCLV::OpenCLVersion platVer;
  ret_err = getPlatformVersion(curPlatform, platVer);

  CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

  pi_result err = PI_SUCCESS;
  if (platVer >= OCLV::V2_1) {

    /* Make sure all devices support CL 2.1 or newer as well. */
    for (cl_device_id dev : devicesInCtx) {
      OCLV::OpenCLVersion devVer;

      ret_err = getDeviceVersion(dev, devVer);
      CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

      /* If the device does not support CL 2.1 or greater, we need to make sure
       * it supports the cl_khr_il_program extension.
       */
      if (devVer < OCLV::V2_1) {
        bool supported = false;

        ret_err = checkDeviceExtensions(dev, {"cl_khr_il_program"}, supported);
        CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

        if (!supported)
          return cast<pi_result>(CL_INVALID_OPERATION);
      }
    }
    if (res_program != nullptr)
      *res_program = cast<pi_program>(clCreateProgramWithIL(
          cast<cl_context>(context), il, length, cast<cl_int *>(&err)));
    return err;
  }

  /* If none of the devices conform with CL 2.1 or newer make sure they all
   * support the cl_khr_il_program extension.
   */
  for (cl_device_id dev : devicesInCtx) {
    bool supported = false;

    ret_err = checkDeviceExtensions(dev, {"cl_khr_il_program"}, supported);
    CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

    if (!supported)
      return cast<pi_result>(CL_INVALID_OPERATION);
  }

  using apiFuncT =
      cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
  apiFuncT funcPtr =
      reinterpret_cast<apiFuncT>(clGetExtensionFunctionAddressForPlatform(
          curPlatform, "clCreateProgramWithILKHR"));

  assert(funcPtr != nullptr);
  if (res_program != nullptr)
    *res_program = cast<pi_program>(
        funcPtr(cast<cl_context>(context), il, length, cast<cl_int *>(&err)));
  else
    err = PI_ERROR_INVALID_VALUE;

  return err;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle nativeHandle,
                                             pi_context, bool,
                                             pi_program *piProgram) {
  assert(piProgram != nullptr);
  *piProgram = reinterpret_cast<pi_program>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piSamplerCreate(pi_context context,
                          const pi_sampler_properties *sampler_properties,
                          pi_sampler *result_sampler) {
  // Initialize properties according to OpenCL 2.1 spec.
  pi_result error_code;
  pi_bool normalizedCoords = PI_TRUE;
  pi_sampler_addressing_mode addressingMode = PI_SAMPLER_ADDRESSING_MODE_CLAMP;
  pi_sampler_filter_mode filterMode = PI_SAMPLER_FILTER_MODE_NEAREST;

  // Unpack sampler properties
  for (std::size_t i = 0; sampler_properties && sampler_properties[i] != 0;
       ++i) {
    if (sampler_properties[i] == PI_SAMPLER_INFO_NORMALIZED_COORDS) {
      normalizedCoords = static_cast<pi_bool>(sampler_properties[++i]);
    } else if (sampler_properties[i] == PI_SAMPLER_INFO_ADDRESSING_MODE) {
      addressingMode =
          static_cast<pi_sampler_addressing_mode>(sampler_properties[++i]);
    } else if (sampler_properties[i] == PI_SAMPLER_INFO_FILTER_MODE) {
      filterMode = static_cast<pi_sampler_filter_mode>(sampler_properties[++i]);
    } else {
      assert(false && "Cannot recognize sampler property");
    }
  }

  // Always call OpenCL 1.0 API
  *result_sampler = cast<pi_sampler>(
      clCreateSampler(cast<cl_context>(context), normalizedCoords,
                      addressingMode, filterMode, cast<cl_int *>(&error_code)));
  return error_code;
}

pi_result piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                                  const pi_mem *arg_value) {
  return cast<pi_result>(
      clSetKernelArg(cast<cl_kernel>(kernel), cast<cl_uint>(arg_index),
                     sizeof(arg_value), cast<const cl_mem *>(arg_value)));
}

pi_result piextKernelSetArgSampler(pi_kernel kernel, pi_uint32 arg_index,
                                   const pi_sampler *arg_value) {
  return cast<pi_result>(
      clSetKernelArg(cast<cl_kernel>(kernel), cast<cl_uint>(arg_index),
                     sizeof(cl_sampler), cast<const cl_sampler *>(arg_value)));
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle nativeHandle,
                                            pi_context, pi_program, bool,
                                            pi_kernel *piKernel) {
  assert(piKernel != nullptr);
  *piKernel = reinterpret_cast<pi_kernel>(nativeHandle);
  return PI_SUCCESS;
}

// Function gets characters between delimeter's in str
// then checks if they are equal to the sub_str.
// returns true if there is at least one instance
// returns false if there are no instances of the name
static bool is_in_separated_string(const std::string &str, char delimiter,
                                   const std::string &sub_str) {
  size_t beg = 0;
  size_t length = 0;
  for (const auto &x : str) {
    if (x == delimiter) {
      if (str.substr(beg, length) == sub_str)
        return true;

      beg += length + 1;
      length = 0;
      continue;
    }
    length++;
  }
  if (length != 0)
    if (str.substr(beg, length) == sub_str)
      return true;

  return false;
}

pi_result piextGetDeviceFunctionPointer(pi_device device, pi_program program,
                                        const char *func_name,
                                        pi_uint64 *function_pointer_ret) {

  cl_context CLContext = nullptr;
  cl_int ret_err =
      clGetProgramInfo(cast<cl_program>(program), CL_PROGRAM_CONTEXT,
                       sizeof(CLContext), &CLContext, nullptr);

  if (ret_err != CL_SUCCESS)
    return cast<pi_result>(ret_err);

  clGetDeviceFunctionPointer_fn FuncT = nullptr;
  ret_err = getExtFuncFromContext<clGetDeviceFunctionPointer_fn>(
      CLContext, ExtFuncPtrCache->clGetDeviceFunctionPointerCache,
      clGetDeviceFunctionPointerName, &FuncT);

  pi_result pi_ret_err = PI_SUCCESS;

  // Check if kernel name exists, to prevent opencl runtime throwing exception
  // with cpu runtime
  // TODO: Use fallback search method if extension does not exist once CPU
  // runtime no longer throws exceptions and prints messages when given
  // unavailable functions.
  *function_pointer_ret = 0;
  size_t Size;
  cl_int Res =
      clGetProgramInfo(cast<cl_program>(program), PI_PROGRAM_INFO_KERNEL_NAMES,
                       0, nullptr, &Size);
  if (Res != CL_SUCCESS)
    return cast<pi_result>(Res);

  std::string ClResult(Size, ' ');
  Res =
      clGetProgramInfo(cast<cl_program>(program), PI_PROGRAM_INFO_KERNEL_NAMES,
                       ClResult.size(), &ClResult[0], nullptr);
  if (Res != CL_SUCCESS)
    return cast<pi_result>(Res);

  // Get rid of the null terminator and search for kernel_name
  // If function cannot be found return error code to indicate it
  // exists
  ClResult.pop_back();
  if (!is_in_separated_string(ClResult, ';', func_name))
    return PI_ERROR_INVALID_KERNEL_NAME;

  pi_ret_err = PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE;

  // If clGetDeviceFunctionPointer is in list of extensions
  if (FuncT) {
    pi_ret_err = cast<pi_result>(FuncT(cast<cl_device_id>(device),
                                       cast<cl_program>(program), func_name,
                                       function_pointer_ret));
    // GPU runtime sometimes returns PI_ERROR_INVALID_ARG_VALUE if func address
    // cannot be found even if kernel exits. As the kernel does exist return
    // that the address is not available
    if (pi_ret_err == CL_INVALID_ARG_VALUE) {
      *function_pointer_ret = 0;
      return PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE;
    }
  }
  return pi_ret_err;
}

pi_result piContextCreate(const pi_context_properties *properties,
                          pi_uint32 num_devices, const pi_device *devices,
                          void (*pfn_notify)(const char *errinfo,
                                             const void *private_info,
                                             size_t cb, void *user_data1),
                          void *user_data, pi_context *retcontext) {
  pi_result ret = PI_ERROR_INVALID_OPERATION;
  *retcontext = cast<pi_context>(
      clCreateContext(properties, cast<cl_uint>(num_devices),
                      cast<const cl_device_id *>(devices), pfn_notify,
                      user_data, cast<cl_int *>(&ret)));

  return ret;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle nativeHandle,
                                             pi_uint32 num_devices,
                                             const pi_device *devices,
                                             bool ownNativeHandle,
                                             pi_context *piContext) {
  (void)num_devices;
  (void)devices;
  (void)ownNativeHandle;
  assert(piContext != nullptr);
  assert(ownNativeHandle == false);
  *piContext = reinterpret_cast<pi_context>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context context, pi_context_info paramName,
                           size_t paramValueSize, void *paramValue,
                           size_t *paramValueSizeRet) {
  switch (paramName) {
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT:
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT: {
    // 2D USM memops are not supported.
    cl_bool result = false;
    std::memcpy(paramValue, &result, sizeof(cl_bool));
    return PI_SUCCESS;
  }
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // These queries should be dealt with in context_impl.cpp by calling the
    // queries of each device separately and building the intersection set.
    setErrorMessage("These queries should have never come here.",
                    PI_ERROR_INVALID_ARG_VALUE);
    return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
  }
  default:
    cl_int result = clGetContextInfo(
        cast<cl_context>(context), cast<cl_context_info>(paramName),
        paramValueSize, paramValue, paramValueSizeRet);
    return static_cast<pi_result>(result);
  }
}

pi_result piMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                            void *host_ptr, pi_mem *ret_mem,
                            const pi_mem_properties *properties) {
  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  if (properties) {
    // TODO: need to check if all properties are supported by OpenCL RT and
    // ignore unsupported
    clCreateBufferWithPropertiesINTEL_fn FuncPtr = nullptr;
    cl_context CLContext = cast<cl_context>(context);
    // First we need to look up the function pointer
    ret_err = getExtFuncFromContext<clCreateBufferWithPropertiesINTEL_fn>(
        CLContext, ExtFuncPtrCache->clCreateBufferWithPropertiesINTELCache,
        clCreateBufferWithPropertiesName, &FuncPtr);
    if (FuncPtr) {
      *ret_mem =
          cast<pi_mem>(FuncPtr(CLContext, properties, cast<cl_mem_flags>(flags),
                               size, host_ptr, cast<cl_int *>(&ret_err)));
      return ret_err;
    }
  }

  *ret_mem = cast<pi_mem>(clCreateBuffer(cast<cl_context>(context),
                                         cast<cl_mem_flags>(flags), size,
                                         host_ptr, cast<cl_int *>(&ret_err)));
  return ret_err;
}

pi_result piMemImageCreate(pi_context context, pi_mem_flags flags,
                           const pi_image_format *image_format,
                           const pi_image_desc *image_desc, void *host_ptr,
                           pi_mem *ret_mem) {
  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_mem = cast<pi_mem>(
      clCreateImage(cast<cl_context>(context), cast<cl_mem_flags>(flags),
                    cast<const cl_image_format *>(image_format),
                    cast<const cl_image_desc *>(image_desc), host_ptr,
                    cast<cl_int *>(&ret_err)));

  return ret_err;
}

pi_result piMemBufferPartition(pi_mem buffer, pi_mem_flags flags,
                               pi_buffer_create_type buffer_create_type,
                               void *buffer_create_info, pi_mem *ret_mem) {

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_mem = cast<pi_mem>(
      clCreateSubBuffer(cast<cl_mem>(buffer), cast<cl_mem_flags>(flags),
                        cast<cl_buffer_create_type>(buffer_create_type),
                        buffer_create_info, cast<cl_int *>(&ret_err)));
  return ret_err;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle nativeHandle,
                                         pi_context context,
                                         bool ownNativeHandle, pi_mem *piMem) {
  (void)context;
  (void)ownNativeHandle;
  assert(piMem != nullptr);
  *piMem = reinterpret_cast<pi_mem>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piextMemImageCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, bool ownNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *Img) {
  (void)context;
  (void)ownNativeHandle;
  (void)ImageFormat;
  (void)ImageDesc;
  assert(Img != nullptr);
  *Img = reinterpret_cast<pi_mem>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithSource(pi_context context, pi_uint32 count,
                                      const char **strings,
                                      const size_t *lengths,
                                      pi_program *ret_program) {

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_program = cast<pi_program>(
      clCreateProgramWithSource(cast<cl_context>(context), cast<cl_uint>(count),
                                strings, lengths, cast<cl_int *>(&ret_err)));
  return ret_err;
}

pi_result piProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *ret_program) {
  (void)metadata;
  (void)num_metadata_entries;

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_program = cast<pi_program>(clCreateProgramWithBinary(
      cast<cl_context>(context), cast<cl_uint>(num_devices),
      cast<const cl_device_id *>(device_list), lengths, binaries,
      cast<cl_int *>(binary_status), cast<cl_int *>(&ret_err)));
  return ret_err;
}

pi_result piProgramLink(pi_context context, pi_uint32 num_devices,
                        const pi_device *device_list, const char *options,
                        pi_uint32 num_input_programs,
                        const pi_program *input_programs,
                        void (*pfn_notify)(pi_program program, void *user_data),
                        void *user_data, pi_program *ret_program) {

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_program = cast<pi_program>(
      clLinkProgram(cast<cl_context>(context), cast<cl_uint>(num_devices),
                    cast<const cl_device_id *>(device_list), options,
                    cast<cl_uint>(num_input_programs),
                    cast<const cl_program *>(input_programs),
                    cast<void (*)(cl_program, void *)>(pfn_notify), user_data,
                    cast<cl_int *>(&ret_err)));
  return ret_err;
}

pi_result piKernelCreate(pi_program program, const char *kernel_name,
                         pi_kernel *ret_kernel) {

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_kernel = cast<pi_kernel>(clCreateKernel(
      cast<cl_program>(program), kernel_name, cast<cl_int *>(&ret_err)));
  return ret_err;
}

pi_result piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                               pi_kernel_group_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  if (kernel == nullptr) {
    return PI_ERROR_INVALID_KERNEL;
  }

  switch (param_name) {
  case PI_KERNEL_GROUP_INFO_NUM_REGS:
    return PI_ERROR_INVALID_VALUE;
  default:
    cl_int result = clGetKernelWorkGroupInfo(
        cast<cl_kernel>(kernel), cast<cl_device_id>(device),
        cast<cl_kernel_work_group_info>(param_name), param_value_size,
        param_value, param_value_size_ret);
    return static_cast<pi_result>(result);
  }
}

pi_result piKernelGetSubGroupInfo(pi_kernel kernel, pi_device device,
                                  pi_kernel_sub_group_info param_name,
                                  size_t input_value_size,
                                  const void *input_value,
                                  size_t param_value_size, void *param_value,
                                  size_t *param_value_size_ret) {
  (void)param_value_size;
  size_t ret_val;
  cl_int ret_err;

  std::shared_ptr<void> implicit_input_value;
  if (param_name == PI_KERNEL_MAX_SUB_GROUP_SIZE && !input_value) {
    // OpenCL needs an input value for PI_KERNEL_MAX_SUB_GROUP_SIZE so if no
    // value is given we use the max work item size of the device in the first
    // dimention to avoid truncation of max sub-group size.
    pi_uint32 max_dims = 0;
    pi_result pi_ret_err =
        piDeviceGetInfo(device, PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(pi_uint32), &max_dims, nullptr);
    if (pi_ret_err != PI_SUCCESS)
      return pi_ret_err;
    std::shared_ptr<size_t[]> WGSizes{new size_t[max_dims]};
    pi_ret_err =
        piDeviceGetInfo(device, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                        max_dims * sizeof(size_t), WGSizes.get(), nullptr);
    if (pi_ret_err != PI_SUCCESS)
      return pi_ret_err;
    for (size_t i = 1; i < max_dims; ++i)
      WGSizes.get()[i] = 1;
    implicit_input_value = std::move(WGSizes);
    input_value_size = max_dims * sizeof(size_t);
    input_value = implicit_input_value.get();
  }

  ret_err = cast<pi_result>(clGetKernelSubGroupInfo(
      cast<cl_kernel>(kernel), cast<cl_device_id>(device),
      cast<cl_kernel_sub_group_info>(param_name), input_value_size, input_value,
      sizeof(size_t), &ret_val, param_value_size_ret));

  if (ret_err == CL_INVALID_OPERATION) {
    // clGetKernelSubGroupInfo returns CL_INVALID_OPERATION if the device does
    // not support subgroups.

    if (param_name == PI_KERNEL_MAX_NUM_SUB_GROUPS) {
      ret_val = 1; // Minimum required by SYCL 2020 spec
      ret_err = CL_SUCCESS;
    } else if (param_name == PI_KERNEL_COMPILE_NUM_SUB_GROUPS) {
      ret_val = 0; // Not specified by kernel
      ret_err = CL_SUCCESS;
    } else if (param_name == PI_KERNEL_MAX_SUB_GROUP_SIZE) {
      // Return the maximum work group size for the kernel
      size_t kernel_work_group_size = 0;
      pi_result pi_ret_err = piKernelGetGroupInfo(
          kernel, device, PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE, sizeof(size_t),
          &kernel_work_group_size, nullptr);
      if (pi_ret_err != PI_SUCCESS)
        return pi_ret_err;
      ret_val = kernel_work_group_size;
      ret_err = CL_SUCCESS;
    } else if (param_name == PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL) {
      ret_val = 0; // Not specified by kernel
      ret_err = CL_SUCCESS;
    }
  }

  if (ret_err != CL_SUCCESS)
    return cast<pi_result>(ret_err);

  *(static_cast<uint32_t *>(param_value)) = static_cast<uint32_t>(ret_val);
  if (param_value_size_ret)
    *param_value_size_ret = sizeof(uint32_t);
  return PI_SUCCESS;
}

pi_result piEventCreate(pi_context context, pi_event *ret_event) {

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  auto *cl_err = cast<cl_int *>(&ret_err);

  cl_event e = clCreateUserEvent(cast<cl_context>(context), cl_err);
  *ret_event = cast<pi_event>(e);
  if (*cl_err != CL_SUCCESS)
    return ret_err;
  *cl_err = clSetUserEventStatus(e, CL_COMPLETE);
  return ret_err;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle nativeHandle,
                                           pi_context context,
                                           bool ownNativeHandle,
                                           pi_event *piEvent) {
  (void)context;
  // TODO: ignore this, but eventually want to return error as unsupported
  (void)ownNativeHandle;

  assert(piEvent != nullptr);
  assert(nativeHandle);
  assert(context);

  *piEvent = reinterpret_cast<pi_event>(nativeHandle);
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferMap(pi_queue command_queue, pi_mem buffer,
                                pi_bool blocking_map, pi_map_flags map_flags,
                                size_t offset, size_t size,
                                pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event, void **ret_map) {

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  *ret_map = cast<void *>(clEnqueueMapBuffer(
      cast<cl_command_queue>(command_queue), cast<cl_mem>(buffer),
      cast<cl_bool>(blocking_map), map_flags, offset, size,
      cast<cl_uint>(num_events_in_wait_list),
      cast<const cl_event *>(event_wait_list), cast<cl_event *>(event),
      cast<cl_int *>(&ret_err)));
  return ret_err;
}

//
// USM
//

/// Allocates host memory accessible by the device.
///
/// \param result_ptr contains the allocated memory
/// \param context is the pi_context
/// \param pi_usm_mem_properties are optional allocation properties
/// \param size_t is the size of the allocation
/// \param alignment is the desired alignment of the allocation
pi_result piextUSMHostAlloc(void **result_ptr, pi_context context,
                            pi_usm_mem_properties *properties, size_t size,
                            pi_uint32 alignment) {

  void *Ptr = nullptr;
  pi_result RetVal = PI_ERROR_INVALID_OPERATION;

  // First we need to look up the function pointer
  clHostMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cast<cl_context>(context);
  RetVal = getExtFuncFromContext<clHostMemAllocINTEL_fn>(
      CLContext, ExtFuncPtrCache->clHostMemAllocINTELCache, clHostMemAllocName,
      &FuncPtr);

  if (FuncPtr) {
    Ptr = FuncPtr(CLContext, cast<cl_mem_properties_intel *>(properties), size,
                  alignment, cast<cl_int *>(&RetVal));
  }

  *result_ptr = Ptr;

  // ensure we aligned the allocation correctly
  if (RetVal == PI_SUCCESS && alignment != 0)
    assert(reinterpret_cast<std::uintptr_t>(*result_ptr) % alignment == 0 &&
           "allocation not aligned correctly");

  return RetVal;
}

/// Allocates device memory
///
/// \param result_ptr contains the allocated memory
/// \param context is the pi_context
/// \param device is the device the memory will be allocated on
/// \param pi_usm_mem_properties are optional allocation properties
/// \param size_t is the size of the allocation
/// \param alignment is the desired alignment of the allocation
pi_result piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                              pi_device device,
                              pi_usm_mem_properties *properties, size_t size,
                              pi_uint32 alignment) {

  void *Ptr = nullptr;
  pi_result RetVal = PI_ERROR_INVALID_OPERATION;

  // First we need to look up the function pointer
  clDeviceMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cast<cl_context>(context);
  RetVal = getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
      CLContext, ExtFuncPtrCache->clDeviceMemAllocINTELCache,
      clDeviceMemAllocName, &FuncPtr);

  if (FuncPtr) {
    Ptr = FuncPtr(CLContext, cast<cl_device_id>(device),
                  cast<cl_mem_properties_intel *>(properties), size, alignment,
                  cast<cl_int *>(&RetVal));
  }

  *result_ptr = Ptr;

  // ensure we aligned the allocation correctly
  if (RetVal == PI_SUCCESS && alignment != 0)
    assert(reinterpret_cast<std::uintptr_t>(*result_ptr) % alignment == 0 &&
           "allocation not aligned correctly");

  return RetVal;
}

/// Allocates memory accessible on both host and device
///
/// \param result_ptr contains the allocated memory
/// \param context is the pi_context
/// \param device is the device the memory will be allocated on
/// \param pi_usm_mem_properties are optional allocation properties
/// \param size_t is the size of the allocation
/// \param alignment is the desired alignment of the allocation
pi_result piextUSMSharedAlloc(void **result_ptr, pi_context context,
                              pi_device device,
                              pi_usm_mem_properties *properties, size_t size,
                              pi_uint32 alignment) {

  void *Ptr = nullptr;
  pi_result RetVal = PI_ERROR_INVALID_OPERATION;

  // First we need to look up the function pointer
  clSharedMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cast<cl_context>(context);
  RetVal = getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
      CLContext, ExtFuncPtrCache->clSharedMemAllocINTELCache,
      clSharedMemAllocName, &FuncPtr);

  if (FuncPtr) {
    Ptr = FuncPtr(cast<cl_context>(context), cast<cl_device_id>(device),
                  cast<cl_mem_properties_intel *>(properties), size, alignment,
                  cast<cl_int *>(&RetVal));
  }

  *result_ptr = Ptr;

  assert(alignment == 0 ||
         (RetVal == PI_SUCCESS &&
          reinterpret_cast<std::uintptr_t>(*result_ptr) % alignment == 0));
  return RetVal;
}

/// Frees allocated USM memory in a blocking manner
///
/// \param context is the pi_context of the allocation
/// \param ptr is the memory to be freed
pi_result piextUSMFree(pi_context context, void *ptr) {
  // Use a blocking free to avoid issues with indirect access from kernels that
  // might be still running.
  clMemBlockingFreeINTEL_fn FuncPtr = nullptr;

  cl_context CLContext = cast<cl_context>(context);
  pi_result RetVal = PI_ERROR_INVALID_OPERATION;
  RetVal = getExtFuncFromContext<clMemBlockingFreeINTEL_fn>(
      CLContext, ExtFuncPtrCache->clMemBlockingFreeINTELCache,
      clMemBlockingFreeName, &FuncPtr);

  if (FuncPtr) {
    RetVal = cast<pi_result>(FuncPtr(CLContext, ptr));
  }

  return RetVal;
}

/// Sets up pointer arguments for CL kernels. An extra indirection
/// is required due to CL argument conventions.
///
/// \param kernel is the kernel to be launched
/// \param arg_index is the index of the kernel argument
/// \param arg_size is the size in bytes of the argument (ignored in CL)
/// \param arg_value is the pointer argument
pi_result piextKernelSetArgPointer(pi_kernel kernel, pi_uint32 arg_index,
                                   size_t arg_size, const void *arg_value) {
  (void)arg_size;

  // Size is unused in CL as pointer args are passed by value.

  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr = clGetKernelInfo(cast<cl_kernel>(kernel), CL_KERNEL_CONTEXT,
                                 sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clSetKernelArgMemPointerINTEL_fn FuncPtr = nullptr;
  pi_result RetVal = getExtFuncFromContext<clSetKernelArgMemPointerINTEL_fn>(
      CLContext, ExtFuncPtrCache->clSetKernelArgMemPointerINTELCache,
      clSetKernelArgMemPointerName, &FuncPtr);

  if (FuncPtr) {
    // OpenCL passes pointers by value not by reference
    // This means we need to deref the arg to get the pointer value
    auto PtrToPtr = reinterpret_cast<const intptr_t *>(arg_value);
    auto DerefPtr = reinterpret_cast<void *>(*PtrToPtr);
    RetVal =
        cast<pi_result>(FuncPtr(cast<cl_kernel>(kernel), arg_index, DerefPtr));
  }

  return RetVal;
}

/// USM Memset API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to memset
/// \param value is value to set. It is interpreted as an 8-bit value and the
///        upper 24 bits are ignored
/// \param count is the size in bytes to memset
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
pi_result piextUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                size_t count, pi_uint32 num_events_in_waitlist,
                                const pi_event *events_waitlist,
                                pi_event *event) {

  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clEnqueueMemsetINTEL_fn FuncPtr = nullptr;
  pi_result RetVal = getExtFuncFromContext<clEnqueueMemsetINTEL_fn>(
      CLContext, ExtFuncPtrCache->clEnqueueMemsetINTELCache,
      clEnqueueMemsetName, &FuncPtr);

  if (FuncPtr) {
    RetVal = cast<pi_result>(FuncPtr(cast<cl_command_queue>(queue), ptr, value,
                                     count, num_events_in_waitlist,
                                     cast<const cl_event *>(events_waitlist),
                                     cast<cl_event *>(event)));
  }

  return RetVal;
}

/// USM Memcpy API
///
/// \param queue is the queue to submit to
/// \param blocking is whether this operation should block the host
/// \param src_ptr is the data to be copied
/// \param dst_ptr is the location the data will be copied
/// \param size is number of bytes to copy
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
pi_result piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking, void *dst_ptr,
                                const void *src_ptr, size_t size,
                                pi_uint32 num_events_in_waitlist,
                                const pi_event *events_waitlist,
                                pi_event *event) {

  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clEnqueueMemcpyINTEL_fn FuncPtr = nullptr;
  pi_result RetVal = getExtFuncFromContext<clEnqueueMemcpyINTEL_fn>(
      CLContext, ExtFuncPtrCache->clEnqueueMemcpyINTELCache,
      clEnqueueMemcpyName, &FuncPtr);

  if (FuncPtr) {
    RetVal = cast<pi_result>(
        FuncPtr(cast<cl_command_queue>(queue), blocking, dst_ptr, src_ptr, size,
                num_events_in_waitlist, cast<const cl_event *>(events_waitlist),
                cast<cl_event *>(event)));
  }

  return RetVal;
}

/// Hint to migrate memory to the device
///
/// \param queue is the queue to submit to
/// \param ptr points to the memory to migrate
/// \param size is the number of bytes to migrate
/// \param flags is a bitfield used to specify memory migration options
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
pi_result piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr, size_t size,
                                  pi_usm_migration_flags flags,
                                  pi_uint32 num_events_in_waitlist,
                                  const pi_event *events_waitlist,
                                  pi_event *event) {
  (void)ptr;
  (void)size;

  // flags is currently unused so fail if set
  if (flags != 0)
    return PI_ERROR_INVALID_VALUE;

  return cast<pi_result>(clEnqueueMarkerWithWaitList(
      cast<cl_command_queue>(queue), num_events_in_waitlist,
      cast<const cl_event *>(events_waitlist), cast<cl_event *>(event)));

  /*
  // Use this once impls support it.
  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clEnqueueMigrateMemINTEL_fn FuncPtr;
  pi_result Err = getExtFuncFromContext<clEnqueueMigrateMemINTEL_fn>(
      cast<pi_context>(CLContext), "clEnqueueMigrateMemINTEL", &FuncPtr);

  if (Err != PI_SUCCESS) {
    RetVal = Err;
  } else {
    RetVal = cast<pi_result>(FuncPtr(
        cast<cl_command_queue>(queue), ptr, size, flags, num_events_in_waitlist,
        reinterpret_cast<const cl_event *>(events_waitlist),
        reinterpret_cast<cl_event *>(event)));
  }
  */
}

/// USM Memadvise API
///
/// \param queue is the queue to submit to
/// \param ptr is the data to be advised
/// \param length is the size in bytes of the meory to advise
/// \param advice is device specific advice
/// \param event is the event that represents this operation
// USM memadvise API to govern behavior of automatic migration mechanisms
pi_result piextUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                   size_t length, pi_mem_advice advice,
                                   pi_event *event) {
  (void)ptr;
  (void)length;
  (void)advice;

  return cast<pi_result>(
      clEnqueueMarkerWithWaitList(cast<cl_command_queue>(queue), 0, nullptr,
                                  reinterpret_cast<cl_event *>(event)));

  /*
  // Change to use this once drivers support it.

  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr = clGetCommandQueueInfo(cast<cl_command_queue>(queue),
                                 CL_QUEUE_CONTEXT,
                                 sizeof(cl_context),
                                 &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clEnqueueMemAdviseINTEL_fn FuncPtr;
  pi_result Err =
    getExtFuncFromContext<clEnqueueMemAdviseINTEL_fn>(
      cast<pi_context>(CLContext), "clEnqueueMemAdviseINTEL", &FuncPtr);

  if (Err != PI_SUCCESS) {
    RetVal = Err;
  } else {
    RetVal = cast<pi_result>(FuncPtr(cast<cl_command_queue>(queue),
                                     ptr, length, advice, 0, nullptr,
                                     reinterpret_cast<cl_event *>(event)));
  }
  */
}

/// USM 2D Fill API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pattern is a pointer with the bytes of the pattern to set
/// \param pattern_size is the size in bytes of the pattern
/// \param pitch is the total width of the destination memory including padding
/// \param width is width in bytes of each row to fill
/// \param height is height the columns to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueFill2D(pi_queue queue, void *ptr,
                                              size_t pitch, size_t pattern_size,
                                              const void *pattern, size_t width,
                                              size_t height,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event) {
  std::ignore = queue;
  std::ignore = ptr;
  std::ignore = pitch;
  std::ignore = pattern_size;
  std::ignore = pattern;
  std::ignore = width;
  std::ignore = height;
  std::ignore = num_events_in_waitlist;
  std::ignore = events_waitlist;
  std::ignore = event;
  return PI_ERROR_INVALID_OPERATION;
}

/// USM 2D Memset API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to memset
/// \param value contains the byte to set with
/// \param pitch is the total width of the destination memory including padding
/// \param width is width in bytes of each row to memset
/// \param height is height the columns to memset
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemset2D(
    pi_queue queue, void *ptr, size_t pitch, int value, size_t width,
    size_t height, pi_uint32 num_events_in_waitlist,
    const pi_event *events_waitlist, pi_event *event) {
  std::ignore = queue;
  std::ignore = ptr;
  std::ignore = pitch;
  std::ignore = value;
  std::ignore = width;
  std::ignore = height;
  std::ignore = num_events_in_waitlist;
  std::ignore = events_waitlist;
  std::ignore = event;
  return PI_ERROR_INVALID_OPERATION;
}

/// USM 2D Memcpy API
///
/// \param queue is the queue to submit to
/// \param blocking is whether this operation should block the host
/// \param dst_ptr is the location the data will be copied
/// \param dst_pitch is the total width of the destination memory including
/// padding
/// \param src_ptr is the data to be copied
/// \param dst_pitch is the total width of the source memory including padding
/// \param width is width in bytes of each row to be copied
/// \param height is height the columns to be copied
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemcpy2D(
    pi_queue queue, pi_bool blocking, void *dst_ptr, size_t dst_pitch,
    const void *src_ptr, size_t src_pitch, size_t width, size_t height,
    pi_uint32 num_events_in_waitlist, const pi_event *events_waitlist,
    pi_event *event) {
  std::ignore = queue;
  std::ignore = blocking;
  std::ignore = dst_ptr;
  std::ignore = dst_pitch;
  std::ignore = src_ptr;
  std::ignore = src_pitch;
  std::ignore = width;
  std::ignore = height;
  std::ignore = num_events_in_waitlist;
  std::ignore = events_waitlist;
  std::ignore = event;
  return PI_ERROR_INVALID_OPERATION;
}

/// API to query information about USM allocated pointers
/// Valid Queries:
///   PI_MEM_ALLOC_TYPE returns host/device/shared pi_host_usm value
///   PI_MEM_ALLOC_BASE_PTR returns the base ptr of an allocation if
///                         the queried pointer fell inside an allocation.
///                         Result must fit in void *
///   PI_MEM_ALLOC_SIZE returns how big the queried pointer's
///                     allocation is in bytes. Result is a size_t.
///   PI_MEM_ALLOC_DEVICE returns the pi_device this was allocated against
///
/// \param context is the pi_context
/// \param ptr is the pointer to query
/// \param param_name is the type of query to perform
/// \param param_value_size is the size of the result in bytes
/// \param param_value is the result
/// \param param_value_ret is how many bytes were written
pi_result piextUSMGetMemAllocInfo(pi_context context, const void *ptr,
                                  pi_mem_alloc_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t *param_value_size_ret) {

  clGetMemAllocInfoINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cast<cl_context>(context);
  pi_result RetVal = getExtFuncFromContext<clGetMemAllocInfoINTEL_fn>(
      CLContext, ExtFuncPtrCache->clGetMemAllocInfoINTELCache,
      clGetMemAllocInfoName, &FuncPtr);

  if (FuncPtr) {
    RetVal = cast<pi_result>(FuncPtr(cast<cl_context>(context), ptr, param_name,
                                     param_value_size, param_value,
                                     param_value_size_ret));
  }

  return RetVal;
}

/// API for writing data from host to a device global variable.
///
/// \param queue is the queue
/// \param program is the program containing the device global variable
/// \param name is the unique identifier for the device global variable
/// \param blocking_write is true if the write should block
/// \param count is the number of bytes to copy
/// \param offset is the byte offset into the device global variable to start
/// copying
/// \param src is a pointer to where the data must be copied from
/// \param num_events_in_wait_list is a number of events in the wait list
/// \param event_wait_list is the wait list
/// \param event is the resulting event
pi_result piextEnqueueDeviceGlobalVariableWrite(
    pi_queue queue, pi_program program, const char *name,
    pi_bool blocking_write, size_t count, size_t offset, const void *src,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  cl_context Ctx = nullptr;
  cl_int Res =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(Ctx), &Ctx, nullptr);

  if (Res != CL_SUCCESS)
    return cast<pi_result>(Res);

  clEnqueueWriteGlobalVariable_fn F = nullptr;
  Res = getExtFuncFromContext<decltype(F)>(
      Ctx, ExtFuncPtrCache->clEnqueueWriteGlobalVariableCache,
      clEnqueueWriteGlobalVariableName, &F);

  if (!F || Res != CL_SUCCESS)
    return PI_ERROR_INVALID_OPERATION;
  Res = F(cast<cl_command_queue>(queue), cast<cl_program>(program), name,
          blocking_write, count, offset, src, num_events_in_wait_list,
          cast<const cl_event *>(event_wait_list), cast<cl_event *>(event));
  return cast<pi_result>(Res);
}

/// API reading data from a device global variable to host.
///
/// \param queue is the queue
/// \param program is the program containing the device global variable
/// \param name is the unique identifier for the device global variable
/// \param blocking_read is true if the read should block
/// \param count is the number of bytes to copy
/// \param offset is the byte offset into the device global variable to start
/// copying
/// \param dst is a pointer to where the data must be copied to
/// \param num_events_in_wait_list is a number of events in the wait list
/// \param event_wait_list is the wait list
/// \param event is the resulting event
pi_result piextEnqueueDeviceGlobalVariableRead(
    pi_queue queue, pi_program program, const char *name, pi_bool blocking_read,
    size_t count, size_t offset, void *dst, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  cl_context Ctx = nullptr;
  cl_int Res =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(Ctx), &Ctx, nullptr);

  if (Res != CL_SUCCESS)
    return cast<pi_result>(Res);

  clEnqueueReadGlobalVariable_fn F = nullptr;
  Res = getExtFuncFromContext<decltype(F)>(
      Ctx, ExtFuncPtrCache->clEnqueueReadGlobalVariableCache,
      clEnqueueReadGlobalVariableName, &F);

  if (!F || Res != CL_SUCCESS)
    return PI_ERROR_INVALID_OPERATION;
  Res = F(cast<cl_command_queue>(queue), cast<cl_program>(program), name,
          blocking_read, count, offset, dst, num_events_in_wait_list,
          cast<const cl_event *>(event_wait_list), cast<cl_event *>(event));
  return cast<pi_result>(Res);
}

pi_result piextEnqueueReadHostPipe(pi_queue queue, pi_program program,
                                   const char *pipe_symbol, pi_bool blocking,
                                   void *ptr, size_t size,
                                   pi_uint32 num_events_in_waitlist,
                                   const pi_event *events_waitlist,
                                   pi_event *event) {
  cl_context CLContext;
  cl_int CLErr =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clEnqueueReadHostPipeINTEL_fn FuncPtr = nullptr;
  pi_result RetVal = getExtFuncFromContext<clEnqueueReadHostPipeINTEL_fn>(
      CLContext, ExtFuncPtrCache->clEnqueueReadHostPipeINTELCache,
      clEnqueueReadHostPipeName, &FuncPtr);

  if (FuncPtr) {
    RetVal = cast<pi_result>(FuncPtr(
        cast<cl_command_queue>(queue), cast<cl_program>(program), pipe_symbol,
        blocking, ptr, size, num_events_in_waitlist,
        cast<const cl_event *>(events_waitlist), cast<cl_event *>(event)));
  }

  return RetVal;
}

pi_result piextEnqueueWriteHostPipe(pi_queue queue, pi_program program,
                                    const char *pipe_symbol, pi_bool blocking,
                                    void *ptr, size_t size,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  cl_context CLContext;
  cl_int CLErr =
      clGetCommandQueueInfo(cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  clEnqueueWriteHostPipeINTEL_fn FuncPtr = nullptr;
  pi_result RetVal = getExtFuncFromContext<clEnqueueWriteHostPipeINTEL_fn>(
      CLContext, ExtFuncPtrCache->clEnqueueWriteHostPipeINTELCache,
      clEnqueueWriteHostPipeName, &FuncPtr);

  if (FuncPtr) {
    RetVal = cast<pi_result>(FuncPtr(
        cast<cl_command_queue>(queue), cast<cl_program>(program), pipe_symbol,
        blocking, ptr, size, num_events_in_waitlist,
        cast<const cl_event *>(events_waitlist), cast<cl_event *>(event)));
  }

  return RetVal;
}

/// API to set attributes controlling kernel execution
///
/// \param kernel is the pi kernel to execute
/// \param param_name is a pi_kernel_exec_info value that specifies the info
///        passed to the kernel
/// \param param_value_size is the size of the value in bytes
/// \param param_value is a pointer to the value to set for the kernel
///
/// If param_name is PI_USM_INDIRECT_ACCESS, the value will be a ptr to
///    the pi_bool value PI_TRUE
/// If param_name is PI_USM_PTRS, the value will be an array of ptrs
pi_result piKernelSetExecInfo(pi_kernel kernel, pi_kernel_exec_info param_name,
                              size_t param_value_size,
                              const void *param_value) {
  if (param_name == PI_USM_INDIRECT_ACCESS &&
      *(static_cast<const pi_bool *>(param_value)) == PI_TRUE) {
    return USMSetIndirectAccess(kernel);
  } else {
    return cast<pi_result>(clSetKernelExecInfo(
        cast<cl_kernel>(kernel), param_name, param_value_size, param_value));
  }
}

pi_result piextProgramSetSpecializationConstant(pi_program prog,
                                                pi_uint32 spec_id,
                                                size_t spec_size,
                                                const void *spec_value) {
  cl_program ClProg = cast<cl_program>(prog);
  cl_context Ctx = nullptr;
  size_t RetSize = 0;
  cl_int Res =
      clGetProgramInfo(ClProg, CL_PROGRAM_CONTEXT, sizeof(Ctx), &Ctx, &RetSize);

  if (Res != CL_SUCCESS)
    return cast<pi_result>(Res);

  clSetProgramSpecializationConstant_fn F = nullptr;
  Res = getExtFuncFromContext<decltype(F)>(
      Ctx, ExtFuncPtrCache->clSetProgramSpecializationConstantCache,
      clSetProgramSpecializationConstantName, &F);

  if (!F || Res != CL_SUCCESS)
    return PI_ERROR_INVALID_OPERATION;
  Res = F(ClProg, spec_id, spec_size, spec_value);
  return cast<pi_result>(Res);
}

/// Common API for getting the native handle of a PI object
///
/// \param piObj is the pi object to get the native handle of
/// \param nativeHandle is a pointer to be set to the native handle
///
/// PI_SUCCESS
static pi_result piextGetNativeHandle(void *piObj,
                                      pi_native_handle *nativeHandle) {
  assert(nativeHandle != nullptr);
  *nativeHandle = reinterpret_cast<pi_native_handle>(piObj);
  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform platform,
                                       pi_native_handle *nativeHandle) {
  return piextGetNativeHandle(platform, nativeHandle);
}

pi_result piextDeviceGetNativeHandle(pi_device device,
                                     pi_native_handle *nativeHandle) {
  return piextGetNativeHandle(device, nativeHandle);
}

pi_result piextContextGetNativeHandle(pi_context context,
                                      pi_native_handle *nativeHandle) {
  return piextGetNativeHandle(context, nativeHandle);
}

pi_result piextQueueGetNativeHandle(pi_queue queue,
                                    pi_native_handle *nativeHandle,
                                    int32_t *nativeHandleDesc) {
  *nativeHandleDesc = 0;
  return piextGetNativeHandle(queue, nativeHandle);
}

pi_result piextMemGetNativeHandle(pi_mem mem, pi_native_handle *nativeHandle) {
  return piextGetNativeHandle(mem, nativeHandle);
}

pi_result piextProgramGetNativeHandle(pi_program program,
                                      pi_native_handle *nativeHandle) {
  return piextGetNativeHandle(program, nativeHandle);
}

pi_result piextKernelGetNativeHandle(pi_kernel kernel,
                                     pi_native_handle *nativeHandle) {
  return piextGetNativeHandle(kernel, nativeHandle);
}

// This API is called by Sycl RT to notify the end of the plugin lifetime.
// Windows: dynamically loaded plugins might have been unloaded already
// when this is called. Sycl RT holds onto the PI plugin so it can be
// called safely. But this is not transitive. If the PI plugin in turn
// dynamically loaded a different DLL, that may have been unloaded.
// TODO: add a global variable lifetime management code here (see
// pi_level_zero.cpp for reference).
pi_result piTearDown(void *PluginParameter) {
  (void)PluginParameter;
  delete ExtFuncPtrCache;
  ExtFuncPtrCache = nullptr;
  return PI_SUCCESS;
}

pi_result piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                  uint64_t *HostTime) {
  OCLV::OpenCLVersion devVer, platVer;
  cl_platform_id platform;
  cl_device_id deviceID = cast<cl_device_id>(Device);

  // TODO: Cache OpenCL version for each device and platform
  auto ret_err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id), &platform, nullptr);
  if (ret_err != CL_SUCCESS) {
    return cast<pi_result>(ret_err);
  }

  ret_err = getDeviceVersion(deviceID, devVer);

  if (ret_err != CL_SUCCESS) {
    return cast<pi_result>(ret_err);
  }

  ret_err = getPlatformVersion(platform, platVer);

  if (platVer < OCLV::V2_1 || devVer < OCLV::V2_1) {
    setErrorMessage(
        "OpenCL version for device and/or platform is less than 2.1",
        PI_ERROR_INVALID_OPERATION);
    return PI_ERROR_INVALID_OPERATION;
  }

  if (DeviceTime) {
    uint64_t dummy;
    clGetDeviceAndHostTimer(deviceID, DeviceTime,
                            HostTime == nullptr ? &dummy : HostTime);

  } else if (HostTime) {
    clGetHostTimer(deviceID, HostTime);
  }

  return PI_SUCCESS;
}

const char SupportedVersion[] = _PI_OPENCL_PLUGIN_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // PI interface supports higher version or the same version.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  if (strlen(SupportedVersion) >= PluginVersionSize)
    return PI_ERROR_INVALID_VALUE;
  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

#define _PI_CL(pi_api, ocl_api)                                                \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&ocl_api);

  // Platform
  _PI_CL(piPlatformsGet, piPlatformsGet)
  _PI_CL(piPlatformGetInfo, piPlatformGetInfo)
  _PI_CL(piextPlatformGetNativeHandle, piextPlatformGetNativeHandle)
  _PI_CL(piextPlatformCreateWithNativeHandle,
         piextPlatformCreateWithNativeHandle)
  // Device
  _PI_CL(piDevicesGet, piDevicesGet)
  _PI_CL(piDeviceGetInfo, piDeviceGetInfo)
  _PI_CL(piDevicePartition, clCreateSubDevices)
  _PI_CL(piDeviceRetain, clRetainDevice)
  _PI_CL(piDeviceRelease, clReleaseDevice)
  _PI_CL(piextDeviceSelectBinary, piextDeviceSelectBinary)
  _PI_CL(piextGetDeviceFunctionPointer, piextGetDeviceFunctionPointer)
  _PI_CL(piextDeviceGetNativeHandle, piextDeviceGetNativeHandle)
  _PI_CL(piextDeviceCreateWithNativeHandle, piextDeviceCreateWithNativeHandle)
  // Context
  _PI_CL(piContextCreate, piContextCreate)
  _PI_CL(piContextGetInfo, piContextGetInfo)
  _PI_CL(piContextRetain, clRetainContext)
  _PI_CL(piContextRelease, clReleaseContext)
  _PI_CL(piextContextGetNativeHandle, piextContextGetNativeHandle)
  _PI_CL(piextContextCreateWithNativeHandle, piextContextCreateWithNativeHandle)
  // Queue
  _PI_CL(piQueueCreate, piQueueCreate)
  _PI_CL(piextQueueCreate, piextQueueCreate)
  _PI_CL(piQueueGetInfo, piQueueGetInfo)
  _PI_CL(piQueueFinish, clFinish)
  _PI_CL(piQueueFlush, clFlush)
  _PI_CL(piQueueRetain, clRetainCommandQueue)
  _PI_CL(piQueueRelease, clReleaseCommandQueue)
  _PI_CL(piextQueueGetNativeHandle, piextQueueGetNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle, piextQueueCreateWithNativeHandle)
  // Memory
  _PI_CL(piMemBufferCreate, piMemBufferCreate)
  _PI_CL(piMemImageCreate, piMemImageCreate)
  _PI_CL(piMemGetInfo, clGetMemObjectInfo)
  _PI_CL(piMemImageGetInfo, clGetImageInfo)
  _PI_CL(piMemRetain, clRetainMemObject)
  _PI_CL(piMemRelease, clReleaseMemObject)
  _PI_CL(piMemBufferPartition, piMemBufferPartition)
  _PI_CL(piextMemGetNativeHandle, piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, piextMemCreateWithNativeHandle)
  // Program
  _PI_CL(piProgramCreate, piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, clGetProgramInfo)
  _PI_CL(piProgramCompile, clCompileProgram)
  _PI_CL(piProgramBuild, clBuildProgram)
  _PI_CL(piProgramLink, piProgramLink)
  _PI_CL(piProgramGetBuildInfo, clGetProgramBuildInfo)
  _PI_CL(piProgramRetain, clRetainProgram)
  _PI_CL(piProgramRelease, clReleaseProgram)
  _PI_CL(piextProgramSetSpecializationConstant,
         piextProgramSetSpecializationConstant)
  _PI_CL(piextProgramGetNativeHandle, piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle, piextProgramCreateWithNativeHandle)
  // Kernel
  _PI_CL(piKernelCreate, piKernelCreate)
  _PI_CL(piKernelSetArg, clSetKernelArg)
  _PI_CL(piKernelGetInfo, clGetKernelInfo)
  _PI_CL(piKernelGetGroupInfo, piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, clRetainKernel)
  _PI_CL(piKernelRelease, clReleaseKernel)
  _PI_CL(piKernelSetExecInfo, piKernelSetExecInfo)
  _PI_CL(piextKernelSetArgPointer, piextKernelSetArgPointer)
  _PI_CL(piextKernelCreateWithNativeHandle, piextKernelCreateWithNativeHandle)
  _PI_CL(piextKernelGetNativeHandle, piextKernelGetNativeHandle)
  // Event
  _PI_CL(piEventCreate, piEventCreate)
  _PI_CL(piEventGetInfo, clGetEventInfo)
  _PI_CL(piEventGetProfilingInfo, clGetEventProfilingInfo)
  _PI_CL(piEventsWait, clWaitForEvents)
  _PI_CL(piEventSetCallback, clSetEventCallback)
  _PI_CL(piEventSetStatus, clSetUserEventStatus)
  _PI_CL(piEventRetain, clRetainEvent)
  _PI_CL(piEventRelease, clReleaseEvent)
  _PI_CL(piextEventGetNativeHandle, piextGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle, piextEventCreateWithNativeHandle)
  // Sampler
  _PI_CL(piSamplerCreate, piSamplerCreate)
  _PI_CL(piSamplerGetInfo, clGetSamplerInfo)
  _PI_CL(piSamplerRetain, clRetainSampler)
  _PI_CL(piSamplerRelease, clReleaseSampler)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, clEnqueueNDRangeKernel)
  _PI_CL(piEnqueueNativeKernel, clEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, clEnqueueMarkerWithWaitList)
  _PI_CL(piEnqueueEventsWaitWithBarrier, clEnqueueBarrierWithWaitList)
  _PI_CL(piEnqueueMemBufferRead, clEnqueueReadBuffer)
  _PI_CL(piEnqueueMemBufferReadRect, clEnqueueReadBufferRect)
  _PI_CL(piEnqueueMemBufferWrite, clEnqueueWriteBuffer)
  _PI_CL(piEnqueueMemBufferWriteRect, clEnqueueWriteBufferRect)
  _PI_CL(piEnqueueMemBufferCopy, clEnqueueCopyBuffer)
  _PI_CL(piEnqueueMemBufferCopyRect, clEnqueueCopyBufferRect)
  _PI_CL(piEnqueueMemBufferFill, clEnqueueFillBuffer)
  _PI_CL(piEnqueueMemImageRead, clEnqueueReadImage)
  _PI_CL(piEnqueueMemImageWrite, clEnqueueWriteImage)
  _PI_CL(piEnqueueMemImageCopy, clEnqueueCopyImage)
  _PI_CL(piEnqueueMemImageFill, clEnqueueFillImage)
  _PI_CL(piEnqueueMemBufferMap, piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, clEnqueueUnmapMemObject)
  // USM
  _PI_CL(piextUSMHostAlloc, piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMEnqueueFill2D, piextUSMEnqueueFill2D)
  _PI_CL(piextUSMEnqueueMemset2D, piextUSMEnqueueMemset2D)
  _PI_CL(piextUSMEnqueueMemcpy2D, piextUSMEnqueueMemcpy2D)
  _PI_CL(piextUSMGetMemAllocInfo, piextUSMGetMemAllocInfo)
  // Device global variable
  _PI_CL(piextEnqueueDeviceGlobalVariableWrite,
         piextEnqueueDeviceGlobalVariableWrite)
  _PI_CL(piextEnqueueDeviceGlobalVariableRead,
         piextEnqueueDeviceGlobalVariableRead)
  // Host Pipe
  _PI_CL(piextEnqueueReadHostPipe, piextEnqueueReadHostPipe)
  _PI_CL(piextEnqueueWriteHostPipe, piextEnqueueWriteHostPipe)

  _PI_CL(piextKernelSetArgMemObj, piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, piextKernelSetArgSampler)
  _PI_CL(piPluginGetLastError, piPluginGetLastError)
  _PI_CL(piTearDown, piTearDown)
  _PI_CL(piGetDeviceAndHostTimer, piGetDeviceAndHostTimer)
  _PI_CL(piPluginGetBackendOption, piPluginGetBackendOption)

#undef _PI_CL

  return PI_SUCCESS;
}

#ifdef _WIN32
#define __SYCL_PLUGIN_DLL_NAME "pi_opencl.dll"
#include "../common_win_pi_trace/common_win_pi_trace.hpp"
#undef __SYCL_PLUGIN_DLL_NAME
#endif

} // end extern 'C'

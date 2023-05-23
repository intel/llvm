//===--------- common.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <climits>
#include <regex>
#include <sycl/detail/cl.h>
#include <ur/ur.hpp>

#include <sycl/detail/pi.h>

#define CL_RETURN_ON_FAILURE(clCall)                                           \
  if (const cl_int cl_result = clCall != CL_SUCCESS) {                         \
    return map_cl_error_to_ur(cl_result);                                      \
  }

namespace OCLV {
class OpenCLVersion {
protected:
  unsigned int ocl_major;
  unsigned int ocl_minor;

public:
  OpenCLVersion() : ocl_major(0), ocl_minor(0) {}

  OpenCLVersion(unsigned int ocl_major, unsigned int ocl_minor)
      : ocl_major(ocl_major), ocl_minor(ocl_minor) {
    if (!isValid()) {
      ocl_major = ocl_minor = 0;
    }
  }

  OpenCLVersion(const char *version) : OpenCLVersion(std::string(version)) {}

  OpenCLVersion(const std::string &version) : ocl_major(0), ocl_minor(0) {
    /* The OpenCL specification defines the full version string as
     * 'OpenCL<space><ocl_major_version.ocl_minor_version><space><platform-specific
     * information>' for platforms and as
     * 'OpenCL<space><ocl_major_version.ocl_minor_version><space><vendor-specific
     * information>' for devices.
     */
    std::regex rx("OpenCL ([0-9]+)\\.([0-9]+)");
    std::smatch match;

    if (std::regex_search(version, match, rx) && (match.size() == 3)) {
      ocl_major = strtoul(match[1].str().c_str(), nullptr, 10);
      ocl_minor = strtoul(match[2].str().c_str(), nullptr, 10);

      if (!isValid()) {
        ocl_major = ocl_minor = 0;
      }
    }
  }

  bool operator==(const OpenCLVersion &v) const {
    return ocl_major == v.ocl_major && ocl_minor == v.ocl_minor;
  }

  bool operator!=(const OpenCLVersion &v) const { return !(*this == v); }

  bool operator<(const OpenCLVersion &v) const {
    if (ocl_major == v.ocl_major)
      return ocl_minor < v.ocl_minor;

    return ocl_major < v.ocl_major;
  }

  bool operator>(const OpenCLVersion &v) const { return v < *this; }

  bool operator<=(const OpenCLVersion &v) const {
    return (*this < v) || (*this == v);
  }

  bool operator>=(const OpenCLVersion &v) const {
    return (*this > v) || (*this == v);
  }

  bool isValid() const {
    switch (ocl_major) {
    case 0:
      return false;
    case 1:
    case 2:
      return ocl_minor <= 2;
    case UINT_MAX:
      return false;
    default:
      return ocl_minor != UINT_MAX;
    }
  }

  int getMajor() const { return ocl_major; }
  int getMinor() const { return ocl_minor; }
};

inline const OpenCLVersion V1_0(1, 0);
inline const OpenCLVersion V1_1(1, 1);
inline const OpenCLVersion V1_2(1, 2);
inline const OpenCLVersion V2_0(2, 0);
inline const OpenCLVersion V2_1(2, 1);
inline const OpenCLVersion V2_2(2, 2);
inline const OpenCLVersion V3_0(3, 0);

} // namespace OCLV

namespace cl {
constexpr size_t MaxMessageSize = 256;
extern thread_local ur_result_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *message,
                                      ur_result_t error_code);

template <class To, class From> To cast(From value) {

  if constexpr (std::is_pointer_v<From>) {
    static_assert(std::is_pointer_v<From> == std::is_pointer_v<To>,
                  "Cast failed pointer check");
    return reinterpret_cast<To>(value);
  } else {
    static_assert(sizeof(From) == sizeof(To), "Cast failed size check");
    static_assert(std::is_signed_v<From> == std::is_signed_v<To>,
                  "Cast failed sign check");
    return static_cast<To>(value);
  }
}
} // namespace cl

ur_result_t map_cl_error_to_ur(cl_int result);

ur_result_t urGetNativeHandle(void *urObj, ur_native_handle_t *nativeHandle);

namespace cl_ext {
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
CONSTFIX char clEnqueueMemFillName[] = "clEnqueueMemFillINTEL";
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

using clEnqueueReadHostPipeINTEL_fn = CL_API_ENTRY
cl_int(CL_API_CALL *)(cl_command_queue queue, cl_program program,
                      const char *pipe_symbol, cl_bool blocking, void *ptr,
                      size_t size, cl_uint num_events_in_waitlist,
                      const cl_event *events_waitlist, cl_event *event);

using clEnqueueWriteHostPipeINTEL_fn = CL_API_ENTRY
cl_int(CL_API_CALL *)(cl_command_queue queue, cl_program program,
                      const char *pipe_symbol, cl_bool blocking, void *ptr,
                      size_t size, cl_uint num_events_in_waitlist,
                      const cl_event *events_waitlist, cl_event *event);

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
  FuncPtrCache<clEnqueueMemFillINTEL_fn> clEnqueueMemFillINTELCache;
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
inline ExtFuncPtrCacheT *ExtFuncPtrCache;

// USM helper function to get an extension function pointer
template <typename T>
static ur_result_t getExtFuncFromContext(cl_context context,
                                         FuncPtrCache<T> &FPtrCache,
                                         const char *FuncName, T *fptr) {
  // TODO
  // Potentially redo caching as UR interface changes.
  // if cached, return cached FuncPtr
  std::lock_guard<std::mutex> CacheLock{FPtrCache.Mutex};
  std::map<cl_context, T> &FPtrMap = FPtrCache.Map;
  auto It = FPtrMap.find(context);
  if (It != FPtrMap.end()) {
    auto F = It->second;
    // if cached that extension is not available return nullptr and
    // UR_RESULT_ERROR_INVALID_VALUE
    *fptr = F;
    return F ? UR_RESULT_SUCCESS : UR_RESULT_ERROR_INVALID_VALUE;
  }

  cl_uint deviceCount;
  cl_int ret_err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
                                    sizeof(cl_uint), &deviceCount, nullptr);

  if (ret_err != CL_SUCCESS || deviceCount < 1) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  std::vector<cl_device_id> devicesInCtx(deviceCount);
  ret_err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             deviceCount * sizeof(cl_device_id),
                             devicesInCtx.data(), nullptr);

  if (ret_err != CL_SUCCESS) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  cl_platform_id curPlatform;
  ret_err = clGetDeviceInfo(devicesInCtx[0], CL_DEVICE_PLATFORM,
                            sizeof(cl_platform_id), &curPlatform, nullptr);

  if (ret_err != CL_SUCCESS) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  T FuncPtr =
      (T)clGetExtensionFunctionAddressForPlatform(curPlatform, FuncName);

  if (!FuncPtr) {
    // Cache that the extension is not available
    FPtrMap[context] = nullptr;
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  *fptr = FuncPtr;
  FPtrMap[context] = FuncPtr;

  return UR_RESULT_SUCCESS;
}
} // namespace cl_ext

//===--------- common.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <climits>
#include <map>
#include <mutex>
#include <regex>
#include <ur/ur.hpp>

/**
 * Call an OpenCL API and, if the result is not CL_SUCCESS, automatically map
 * the OpenCL error to UR and return from the current function.
 */
#define CL_RETURN_ON_FAILURE(clCall)                                           \
  if (const cl_int cl_result_macro = clCall; cl_result_macro != CL_SUCCESS) {  \
    return mapCLErrorToUR(cl_result_macro);                                    \
  }

/**
 * Call an UR API and, if the result is not UR_RESULT_SUCCESS, automatically
 * return from the current function.
 */
#define UR_RETURN_ON_FAILURE(urCall)                                           \
  if (const ur_result_t ur_result_macro = urCall;                              \
      ur_result_macro != UR_RESULT_SUCCESS) {                                  \
    return ur_result_macro;                                                    \
  }

/**
 * Call an OpenCL API and, if the result is not CL_SUCCESS, automatically return
 * from the current function and set the pointer `outPtr` to nullptr. The OpenCL
 * error is mapped to UR
 */
#define CL_RETURN_ON_FAILURE_AND_SET_NULL(clCall, outPtr)                      \
  if (const cl_int cl_result_macro = clCall != CL_SUCCESS) {                   \
    if (outPtr != nullptr) {                                                   \
      *outPtr = nullptr;                                                       \
    }                                                                          \
    return mapCLErrorToUR(cl_result_macro);                                    \
  }

namespace oclv {
class OpenCLVersion {
protected:
  unsigned int OCLMajor;
  unsigned int OCLMinor;

public:
  OpenCLVersion() : OCLMajor(0), OCLMinor(0) {}

  OpenCLVersion(unsigned int OclMajor, unsigned int OclMinor)
      : OCLMajor(OclMajor), OCLMinor(OclMinor) {
    if (!isValid()) {
      OclMajor = OclMinor = 0;
    }
  }

  OpenCLVersion(const char *Version) : OpenCLVersion(std::string(Version)) {}

  OpenCLVersion(const std::string &Version) : OCLMajor(0), OCLMinor(0) {
    /* The OpenCL specification defines the full version string as
     * 'OpenCL<space><ocl_major_version.ocl_minor_version><space><platform-specific
     * information>' for platforms and as
     * 'OpenCL<space><ocl_major_version.ocl_minor_version><space><vendor-specific
     * information>' for devices.
     */
    std::regex Rx("OpenCL ([0-9]+)\\.([0-9]+)");
    std::smatch Match;

    if (std::regex_search(Version, Match, Rx) && (Match.size() == 3)) {
      OCLMajor = strtoul(Match[1].str().c_str(), nullptr, 10);
      OCLMinor = strtoul(Match[2].str().c_str(), nullptr, 10);

      if (!isValid()) {
        OCLMajor = OCLMinor = 0;
      }
    }
  }

  bool operator==(const OpenCLVersion &V) const {
    return OCLMajor == V.OCLMajor && OCLMinor == V.OCLMinor;
  }

  bool operator!=(const OpenCLVersion &V) const { return !(*this == V); }

  bool operator<(const OpenCLVersion &V) const {
    if (OCLMajor == V.OCLMajor)
      return OCLMinor < V.OCLMinor;

    return OCLMajor < V.OCLMajor;
  }

  bool operator>(const OpenCLVersion &V) const { return V < *this; }

  bool operator<=(const OpenCLVersion &V) const {
    return (*this < V) || (*this == V);
  }

  bool operator>=(const OpenCLVersion &V) const {
    return (*this > V) || (*this == V);
  }

  bool isValid() const {
    switch (OCLMajor) {
    case 0:
      return false;
    case 1:
    case 2:
      return OCLMinor <= 2;
    case UINT_MAX:
      return false;
    default:
      return OCLMinor != UINT_MAX;
    }
  }

  unsigned int getMajor() const { return OCLMajor; }
  unsigned int getMinor() const { return OCLMinor; }
};

inline const OpenCLVersion V1_0(1, 0);
inline const OpenCLVersion V1_1(1, 1);
inline const OpenCLVersion V1_2(1, 2);
inline const OpenCLVersion V2_0(2, 0);
inline const OpenCLVersion V2_1(2, 1);
inline const OpenCLVersion V2_2(2, 2);
inline const OpenCLVersion V3_0(3, 0);

} // namespace oclv

namespace cl_adapter {
constexpr size_t MaxMessageSize = 256;
extern thread_local int32_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *Message,
                                      ur_result_t ErrorCode);

[[noreturn]] void die(const char *Message);

template <class To, class From> To cast(From Value) {

  if constexpr (std::is_pointer_v<From>) {
    static_assert(std::is_pointer_v<From> == std::is_pointer_v<To>,
                  "Cast failed pointer check");
    return reinterpret_cast<To>(Value);
  } else {
    static_assert(sizeof(From) == sizeof(To), "Cast failed size check");
    static_assert(std::is_signed_v<From> == std::is_signed_v<To>,
                  "Cast failed sign check");
    return static_cast<To>(Value);
  }
}
} // namespace cl_adapter

namespace cl_ext {
// Older versions of GCC don't like "const" here
#if defined(__GNUC__) && (__GNUC__ < 7 || (__GNU__C == 7 && __GNUC_MINOR__ < 2))
#define CONSTFIX constexpr
#else
#define CONSTFIX const
#endif

// Names of USM functions that are queried from OpenCL
CONSTFIX char HostMemAllocName[] = "clHostMemAllocINTEL";
CONSTFIX char DeviceMemAllocName[] = "clDeviceMemAllocINTEL";
CONSTFIX char SharedMemAllocName[] = "clSharedMemAllocINTEL";
CONSTFIX char MemBlockingFreeName[] = "clMemBlockingFreeINTEL";
CONSTFIX char CreateBufferWithPropertiesName[] =
    "clCreateBufferWithPropertiesINTEL";
CONSTFIX char SetKernelArgMemPointerName[] = "clSetKernelArgMemPointerINTEL";
CONSTFIX char EnqueueMemFillName[] = "clEnqueueMemFillINTEL";
CONSTFIX char EnqueueMemcpyName[] = "clEnqueueMemcpyINTEL";
CONSTFIX char GetMemAllocInfoName[] = "clGetMemAllocInfoINTEL";
CONSTFIX char SetProgramSpecializationConstantName[] =
    "clSetProgramSpecializationConstant";
CONSTFIX char GetDeviceFunctionPointerName[] =
    "clGetDeviceFunctionPointerINTEL";
CONSTFIX char EnqueueWriteGlobalVariableName[] =
    "clEnqueueWriteGlobalVariableINTEL";
CONSTFIX char EnqueueReadGlobalVariableName[] =
    "clEnqueueReadGlobalVariableINTEL";
// Names of host pipe functions queried from OpenCL
CONSTFIX char EnqueueReadHostPipeName[] = "clEnqueueReadHostPipeINTEL";
CONSTFIX char EnqueueWriteHostPipeName[] = "clEnqueueWriteHostPipeINTEL";

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
                      const char *pipe_symbol, cl_bool blocking,
                      const void *ptr, size_t size,
                      cl_uint num_events_in_waitlist,
                      const cl_event *events_waitlist, cl_event *event);

template <typename T> struct FuncPtrCache {
  std::map<cl_context, T> Map;
  std::mutex Mutex;
};

// FIXME: There's currently no mechanism for cleaning up this cache, meaning
// that it is invalidated whenever a context is destroyed. This could lead to
// reusing an invalid function pointer if another context happens to have the
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
inline std::unique_ptr<ExtFuncPtrCacheT> ExtFuncPtrCache;

// USM helper function to get an extension function pointer
template <typename T>
static ur_result_t getExtFuncFromContext(cl_context Context,
                                         FuncPtrCache<T> &FPtrCache,
                                         const char *FuncName, T *Fptr) {
  // TODO
  // Potentially redo caching as UR interface changes.
  // if cached, return cached FuncPtr
  std::lock_guard<std::mutex> CacheLock{FPtrCache.Mutex};
  std::map<cl_context, T> &FPtrMap = FPtrCache.Map;
  auto It = FPtrMap.find(Context);
  if (It != FPtrMap.end()) {
    auto F = It->second;
    // if cached that extension is not available return nullptr and
    // UR_RESULT_ERROR_INVALID_VALUE
    *Fptr = F;
    return F ? UR_RESULT_SUCCESS : UR_RESULT_ERROR_INVALID_VALUE;
  }

  cl_uint DeviceCount;
  cl_int RetErr = clGetContextInfo(Context, CL_CONTEXT_NUM_DEVICES,
                                   sizeof(cl_uint), &DeviceCount, nullptr);

  if (RetErr != CL_SUCCESS || DeviceCount < 1) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  std::vector<cl_device_id> DevicesInCtx(DeviceCount);
  RetErr = clGetContextInfo(Context, CL_CONTEXT_DEVICES,
                            DeviceCount * sizeof(cl_device_id),
                            DevicesInCtx.data(), nullptr);

  if (RetErr != CL_SUCCESS) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  cl_platform_id CurPlatform;
  RetErr = clGetDeviceInfo(DevicesInCtx[0], CL_DEVICE_PLATFORM,
                           sizeof(cl_platform_id), &CurPlatform, nullptr);

  if (RetErr != CL_SUCCESS) {
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  }

  T FuncPtr = reinterpret_cast<T>(
      clGetExtensionFunctionAddressForPlatform(CurPlatform, FuncName));

  if (!FuncPtr) {
    // Cache that the extension is not available
    FPtrMap[Context] = nullptr;
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  *Fptr = FuncPtr;
  FPtrMap[Context] = FuncPtr;

  return UR_RESULT_SUCCESS;
}
} // namespace cl_ext

ur_result_t mapCLErrorToUR(cl_int Result);

ur_result_t getNativeHandle(void *URObj, ur_native_handle_t *NativeHandle);

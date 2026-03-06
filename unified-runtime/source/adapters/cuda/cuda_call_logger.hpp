//===--------- cuda_call_logger.hpp - CUDA API Call Logger ---------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "logger/ur_logger.hpp"
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sstream>

namespace ur::cuda::call_logger {

inline bool isEnabled() {
  static bool enabled = [] {
    const char *env = std::getenv("UR_CUDA_CALL_TRACE");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

// Helper to format pointer addresses
template <typename T> std::string formatPtr(T *ptr) {
  std::stringstream ss;
  ss << std::hex << "0x" << reinterpret_cast<uintptr_t>(ptr);
  return ss.str();
}

// Helper to format CUdeviceptr
inline std::string formatDevPtr(CUdeviceptr ptr) {
  std::stringstream ss;
  ss << std::hex << "0x" << ptr;
  return ss.str();
}

// Helper to format size_t
inline std::string formatSize(size_t size) {
  std::stringstream ss;
  ss << size << " bytes";
  return ss.str();
}

// Context logging
inline void logContextSwitch(const char *operation, CUcontext ctx,
                             const char *func, int line) {
  if (!isEnabled())
    return;
  std::cerr << "[CUDA] " << func << ":" << line << " " << operation
            << " ctx=" << formatPtr(ctx) << "\n";
}

// Memory operation logging
inline void logMemcpy(const char *operation, CUdeviceptr dst, CUdeviceptr src,
                      size_t size, CUstream stream, const char *func,
                      int line) {
  if (!isEnabled())
    return;
  std::cerr << "[CUDA] " << func << ":" << line << " " << operation
            << " dst=" << formatDevPtr(dst) << " src=" << formatDevPtr(src)
            << " size=" << formatSize(size) << " stream=" << formatPtr(stream)
            << "\n";
}

// Kernel launch logging
inline void logKernelLaunch(CUfunction kernel, unsigned gridX, unsigned gridY,
                            unsigned gridZ, unsigned blockX, unsigned blockY,
                            unsigned blockZ, unsigned sharedMem,
                            CUstream stream, const char *func, int line) {
  if (!isEnabled())
    return;
  std::cerr << "[CUDA] " << func << ":" << line << " cuLaunchKernel"
            << " kernel=" << formatPtr(kernel) << " grid=(" << gridX << ","
            << gridY << "," << gridZ << ") block=(" << blockX << "," << blockY
            << "," << blockZ << ") sharedMem=" << sharedMem
            << " stream=" << formatPtr(stream) << "\n";
}

// Generic CUDA call logging
inline void logCudaCall(const char *callString, const char *func, int line) {
  if (!isEnabled())
    return;
  std::cerr << "[CUDA] " << func << ":" << line << " " << callString << "\n";
}

inline void logCudaResult(CUresult result, const char *func, int line) {
  if (!isEnabled())
    return;
  if (result == CUDA_SUCCESS) {
    std::cerr << "[CUDA] " << func << ":" << line << " -> CUDA_SUCCESS\n";
  } else {
    const char *errStr = nullptr;
    cuGetErrorString(result, &errStr);
    std::cerr << "[CUDA] " << func << ":" << line << " -> ERROR: " << result
              << " (" << (errStr ? errStr : "unknown") << ")\n";
  }
}

// Overloads for other return types (no-op, we only log CUresult)
inline void logCudaResult(nvmlReturn_t, const char *, int) {}
inline void logCudaResult(ur_result_t, const char *, int) {}

} // namespace ur::cuda::call_logger

// Macros for specific CUDA API calls with detailed logging
#define CUDA_CALL_TRACE_CTX_SET(ctx)                                           \
  do {                                                                         \
    ::ur::cuda::call_logger::logContextSwitch("cuCtxSetCurrent", ctx,          \
                                              __func__, __LINE__);             \
  } while (0)

#define CUDA_CALL_TRACE_CTX_GET(ctx)                                           \
  do {                                                                         \
    ::ur::cuda::call_logger::logContextSwitch("cuCtxGetCurrent", *ctx,         \
                                              __func__, __LINE__);             \
  } while (0)

#define CUDA_CALL_TRACE_MEMCPY_ASYNC(dst, src, size, stream)                   \
  do {                                                                         \
    ::ur::cuda::call_logger::logMemcpy("cuMemcpyDtoDAsync", dst, src, size,    \
                                       stream, __func__, __LINE__);            \
  } while (0)

#define CUDA_CALL_TRACE_MEMCPY_GENERIC(dst, src, size, stream)                 \
  do {                                                                         \
    ::ur::cuda::call_logger::logMemcpy("cuMemcpyAsync", dst, src, size,        \
                                       stream, __func__, __LINE__);            \
  } while (0)

#define CUDA_CALL_TRACE_KERNEL_LAUNCH(kernel, gx, gy, gz, bx, by, bz, sm,      \
                                      stream)                                  \
  do {                                                                         \
    ::ur::cuda::call_logger::logKernelLaunch(kernel, gx, gy, gz, bx, by, bz,   \
                                             sm, stream, __func__, __LINE__);  \
  } while (0)

#define CUDA_CALL_TRACE_GENERIC(call_str)                                      \
  do {                                                                         \
    ::ur::cuda::call_logger::logCudaCall(call_str, __func__, __LINE__);        \
  } while (0)

#define CUDA_CALL_TRACE_RESULT(result)                                         \
  do {                                                                         \
    ::ur::cuda::call_logger::logCudaResult(result, __func__, __LINE__);        \
  } while (0)

// Wrapper for UR_CHECK_ERROR that adds logging
#define UR_CHECK_ERROR_TRACED(result)                                          \
  do {                                                                         \
    CUresult __result = (result);                                              \
    CUDA_CALL_TRACE_RESULT(__result);                                          \
    UR_CHECK_ERROR(__result);                                                  \
  } while (0)

// Universal wrapper that logs the CUDA call before executing it
#define UR_CUDA_CALL_LOGGED(call)                                              \
  do {                                                                         \
    CUDA_CALL_TRACE_GENERIC(#call);                                            \
    UR_CHECK_ERROR(call);                                                      \
  } while (0)

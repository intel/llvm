//===--------- command_buffer.hpp - CUDA Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur/ur.hpp>
#include <ur_api.h>

#include "context.hpp"
#include <cuda.h>
#include <memory>

static auto getUrResultString = [](ur_result_t Result) {
  switch (Result) {
  case UR_RESULT_SUCCESS:
    return "UR_RESULT_SUCCESS";
  case UR_RESULT_ERROR_INVALID_OPERATION:
    return "UR_RESULT_ERROR_INVALID_OPERATION";
  case UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES:
    return "UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES";
  case UR_RESULT_ERROR_INVALID_QUEUE:
    return "UR_RESULT_ERROR_INVALID_QUEUE";
  case UR_RESULT_ERROR_INVALID_VALUE:
    return "UR_RESULT_ERROR_INVALID_VALUE";
  case UR_RESULT_ERROR_INVALID_CONTEXT:
    return "UR_RESULT_ERROR_INVALID_CONTEXT";
  case UR_RESULT_ERROR_INVALID_PLATFORM:
    return "UR_RESULT_ERROR_INVALID_PLATFORM";
  case UR_RESULT_ERROR_INVALID_BINARY:
    return "UR_RESULT_ERROR_INVALID_BINARY";
  case UR_RESULT_ERROR_INVALID_PROGRAM:
    return "UR_RESULT_ERROR_INVALID_PROGRAM";
  case UR_RESULT_ERROR_INVALID_SAMPLER:
    return "UR_RESULT_ERROR_INVALID_SAMPLER";
  case UR_RESULT_ERROR_INVALID_BUFFER_SIZE:
    return "UR_RESULT_ERROR_INVALID_BUFFER_SIZE";
  case UR_RESULT_ERROR_INVALID_MEM_OBJECT:
    return "UR_RESULT_ERROR_INVALID_MEM_OBJECT";
  case UR_RESULT_ERROR_INVALID_EVENT:
    return "UR_RESULT_ERROR_INVALID_EVENT";
  case UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST:
    return "UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST";
  case UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
    return "UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET";
  case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
    return "UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE";
  case UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE:
    return "UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE";
  case UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE:
    return "UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE";
  case UR_RESULT_ERROR_DEVICE_NOT_FOUND:
    return "UR_RESULT_ERROR_DEVICE_NOT_FOUND";
  case UR_RESULT_ERROR_INVALID_DEVICE:
    return "UR_RESULT_ERROR_INVALID_DEVICE";
  case UR_RESULT_ERROR_DEVICE_LOST:
    return "UR_RESULT_ERROR_DEVICE_LOST";
  case UR_RESULT_ERROR_DEVICE_REQUIRES_RESET:
    return "UR_RESULT_ERROR_DEVICE_REQUIRES_RESET";
  case UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
    return "UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
  case UR_RESULT_ERROR_DEVICE_PARTITION_FAILED:
    return "UR_RESULT_ERROR_DEVICE_PARTITION_FAILED";
  case UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT:
    return "UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT";
  case UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE:
    return "UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE";
  case UR_RESULT_ERROR_INVALID_WORK_DIMENSION:
    return "UR_RESULT_ERROR_INVALID_WORK_DIMENSION";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGS:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ARGS";
  case UR_RESULT_ERROR_INVALID_KERNEL:
    return "UR_RESULT_ERROR_INVALID_KERNEL";
  case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "UR_RESULT_ERROR_INVALID_KERNEL_NAME";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  case UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  case UR_RESULT_ERROR_INVALID_IMAGE_SIZE:
    return "UR_RESULT_ERROR_INVALID_IMAGE_SIZE";
  case UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
    return "UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED";
  case UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
    return "UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE";
  case UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE:
    return "UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE";
  case UR_RESULT_ERROR_UNINITIALIZED:
    return "UR_RESULT_ERROR_UNINITIALIZED";
  case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "UR_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case UR_RESULT_ERROR_OUT_OF_RESOURCES:
    return "UR_RESULT_ERROR_OUT_OF_RESOURCES";
  case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
    return "UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE";
  case UR_RESULT_ERROR_PROGRAM_LINK_FAILURE:
    return "UR_RESULT_ERROR_PROGRAM_LINK_FAILURE";
  case UR_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "UR_RESULT_ERROR_UNSUPPORTED_VERSION";
  case UR_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "UR_RESULT_ERROR_UNSUPPORTED_FEATURE";
  case UR_RESULT_ERROR_INVALID_ARGUMENT:
    return "UR_RESULT_ERROR_INVALID_ARGUMENT";
  case UR_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "UR_RESULT_ERROR_INVALID_NULL_HANDLE";
  case UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  case UR_RESULT_ERROR_INVALID_NULL_POINTER:
    return "UR_RESULT_ERROR_INVALID_NULL_POINTER";
  case UR_RESULT_ERROR_INVALID_SIZE:
    return "UR_RESULT_ERROR_INVALID_SIZE";
  case UR_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "UR_RESULT_ERROR_UNSUPPORTED_SIZE";
  case UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  case UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  case UR_RESULT_ERROR_INVALID_ENUMERATION:
    return "UR_RESULT_ERROR_INVALID_ENUMERATION";
  case UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "UR_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "UR_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case UR_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return "UR_RESULT_ERROR_INVALID_FUNCTION_NAME";
  case UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  case UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  case UR_RESULT_ERROR_PROGRAM_UNLINKED:
    return "UR_RESULT_ERROR_PROGRAM_UNLINKED";
  case UR_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "UR_RESULT_ERROR_OVERLAPPING_REGIONS";
  case UR_RESULT_ERROR_INVALID_HOST_PTR:
    return "UR_RESULT_ERROR_INVALID_HOST_PTR";
  case UR_RESULT_ERROR_INVALID_USM_SIZE:
    return "UR_RESULT_ERROR_INVALID_USM_SIZE";
  case UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE:
    return "UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE";
  case UR_RESULT_ERROR_ADAPTER_SPECIFIC:
    return "UR_RESULT_ERROR_ADAPTER_SPECIFIC";
  default:
    return "UR_RESULT_ERROR_UNKNOWN";
  }
};

// Trace an internal PI call; returns in case of an error.
#define UR_CALL(Call)                                                          \
  {                                                                            \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR ---> %s\n", #Call);                                  \
    ur_result_t Result = (Call);                                               \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR <--- %s(%s)\n", #Call, getUrResultString(Result));   \
    if (Result != UR_RESULT_SUCCESS)                                           \
      return Result;                                                           \
  }

struct ur_exp_command_buffer_handle_t_ {

  ur_exp_command_buffer_handle_t_(ur_context_handle_t hContext,
                                  ur_device_handle_t hDevice);

  ~ur_exp_command_buffer_handle_t_();

  void RegisterSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint,
                         std::shared_ptr<CUgraphNode> CuNode) {
    SyncPoints[SyncPoint] = CuNode;
    NextSyncPoint++;
  }

  ur_exp_command_buffer_sync_point_t GetNextSyncPoint() const {
    return NextSyncPoint;
  }

  // Helper to register next sync point
  // @param CuNode Node to register as next sycn point
  // @return Pointer to the sync that registers the Node
  ur_exp_command_buffer_sync_point_t
  AddSyncPoint(std::shared_ptr<CUgraphNode> CuNode) {
    ur_exp_command_buffer_sync_point_t SyncPoint = NextSyncPoint;
    RegisterSyncPoint(SyncPoint, CuNode);
    return SyncPoint;
  }

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command buffer
  ur_device_handle_t Device;
  // Cuda Graph handle
  CUgraph CudaGraph;
  // Cuda Graph Exec handle
  CUgraphExec CudaGraphExec;
  // Atomic variable counting the number of reference to this command_buffer
  // using std::atomic prevents data race when incrementing/decrementing.
  std::atomic_uint32_t RefCount;

  // Map of sync_points to ur_events
  std::unordered_map<ur_exp_command_buffer_sync_point_t,
                     std::shared_ptr<CUgraphNode>>
      SyncPoints;
  // Next sync_point value (may need to consider ways to reuse values if 32-bits
  // is not enough)
  ur_exp_command_buffer_sync_point_t NextSyncPoint;

  // Used when retaining an object.
  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }
  // Used when releasing an object.
  uint32_t decrementReferenceCount() noexcept { return --RefCount; }
  uint32_t getReferenceCount() const noexcept { return RefCount; }
  // This method allows to guard a code which needs to be executed when object's
  // ref count becomes zero after release. It is important to notice that only a
  // single thread can pass through this check. This is true because of several
  // reasons:
  //   1. Decrement operation is executed atomically.
  //   2. It is not allowed to retain an object after its refcount reaches zero.
  //   3. It is not allowed to release an object more times than the value of
  //   the ref count.
  // 2. and 3. basically means that we can't use an object at all as soon as its
  // refcount reaches zero. Using this check guarantees that code for deleting
  // an object and releasing its resources is executed once by a single thread
  // and we don't need to use any mutexes to guard access to this object in the
  // scope after this check. Of course if we access another objects in this code
  // (not the one which is being deleted) then access to these objects must be
  // guarded, for example with a mutex.
  bool decrementAndTestReferenceCount() { return --RefCount == 0; }
};

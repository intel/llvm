//==---------- pi.h - Plugin Interface -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi The Plugin Interface
// TODO: link to sphinx page

/// \file Main Plugin Interface header file.
///
/// This is the definition of a generic offload Plugin Interface (PI), which is
/// used by the SYCL implementation to connect to multiple device back-ends,
/// e.g. to OpenCL. The interface is intentionally kept C-only for the
/// purpose of having full flexibility and interoperability with different
/// environments.
///
/// \ingroup sycl_pi

#ifndef _PI_H_
#define _PI_H_

// Every single change in PI API should be accompanied with the minor
// version increase (+1). In the cases where backward compatibility is not
// maintained there should be a (+1) change to the major version in
// addition to the increase of the minor.
//
// PI version changes log:
// -- Version 1.2:
// 1. (Binary backward compatibility breaks) Two fields added to the
// pi_device_binary_struct structure:
//   pi_device_binary_property_set PropertySetsBegin;
//   pi_device_binary_property_set PropertySetsEnd;
// 2. A number of types needed to define pi_device_binary_property_set added.
// 3. Added new ownership argument to piextContextCreateWithNativeHandle.
// 4. Add interoperability interfaces for kernel.
// 4.6 Added new ownership argument to piextQueueCreateWithNativeHandle which
// changes the API version from 3.5 to 4.6.
// 5.7 Added new context and ownership arguments to
//   piextEventCreateWithNativeHandle
// 6.8 Added new ownership argument to piextProgramCreateWithNativeHandle. Added
// piQueueFlush function.
//
#include "CL/cl.h"
#define _PI_H_VERSION_MAJOR 6
#define _PI_H_VERSION_MINOR 8

#define _PI_STRING_HELPER(a) #a
#define _PI_CONCAT(a, b) _PI_STRING_HELPER(a.b)
#define _PI_H_VERSION_STRING                                                   \
  _PI_CONCAT(_PI_H_VERSION_MAJOR, _PI_H_VERSION_MINOR)

// TODO: we need a mapping of PI to OpenCL somewhere, and this can be done
// elsewhere, e.g. in the pi_opencl, but constants/enums mapping is now
// done here, for efficiency and simplicity.
#include <CL/cl_ext.h>
#include <CL/sycl/detail/cl.h>
#include <CL/sycl/detail/export.hpp>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

using pi_int32 = int32_t;
using pi_uint32 = uint32_t;
using pi_uint64 = uint64_t;
using pi_bool = pi_uint32;
using pi_bitfield = pi_uint64;
using pi_native_handle = uintptr_t;

//
// NOTE: prefer to map 1:1 to OpenCL so that no translation is needed
// for PI <-> OpenCL ways. The PI <-> to other BE translation is almost
// always needed anyway.
//
// TODO: populate PI enums.
//
typedef enum {
  PI_SUCCESS = CL_SUCCESS,
  PI_INVALID_KERNEL_NAME = CL_INVALID_KERNEL_NAME,
  PI_INVALID_OPERATION = CL_INVALID_OPERATION,
  PI_INVALID_KERNEL = CL_INVALID_KERNEL,
  PI_INVALID_QUEUE_PROPERTIES = CL_INVALID_QUEUE_PROPERTIES,
  PI_INVALID_VALUE = CL_INVALID_VALUE,
  PI_INVALID_CONTEXT = CL_INVALID_CONTEXT,
  PI_INVALID_PLATFORM = CL_INVALID_PLATFORM,
  PI_INVALID_DEVICE = CL_INVALID_DEVICE,
  PI_INVALID_BINARY = CL_INVALID_BINARY,
  PI_INVALID_QUEUE = CL_INVALID_COMMAND_QUEUE,
  PI_OUT_OF_HOST_MEMORY = CL_OUT_OF_HOST_MEMORY,
  PI_INVALID_PROGRAM = CL_INVALID_PROGRAM,
  PI_INVALID_PROGRAM_EXECUTABLE = CL_INVALID_PROGRAM_EXECUTABLE,
  PI_INVALID_SAMPLER = CL_INVALID_SAMPLER,
  PI_INVALID_BUFFER_SIZE = CL_INVALID_BUFFER_SIZE,
  PI_INVALID_MEM_OBJECT = CL_INVALID_MEM_OBJECT,
  PI_OUT_OF_RESOURCES = CL_OUT_OF_RESOURCES,
  PI_INVALID_EVENT = CL_INVALID_EVENT,
  PI_INVALID_EVENT_WAIT_LIST = CL_INVALID_EVENT_WAIT_LIST,
  PI_MISALIGNED_SUB_BUFFER_OFFSET = CL_MISALIGNED_SUB_BUFFER_OFFSET,
  PI_BUILD_PROGRAM_FAILURE = CL_BUILD_PROGRAM_FAILURE,
  PI_INVALID_WORK_GROUP_SIZE = CL_INVALID_WORK_GROUP_SIZE,
  PI_COMPILER_NOT_AVAILABLE = CL_COMPILER_NOT_AVAILABLE,
  PI_PROFILING_INFO_NOT_AVAILABLE = CL_PROFILING_INFO_NOT_AVAILABLE,
  PI_DEVICE_NOT_FOUND = CL_DEVICE_NOT_FOUND,
  PI_INVALID_WORK_ITEM_SIZE = CL_INVALID_WORK_ITEM_SIZE,
  PI_INVALID_WORK_DIMENSION = CL_INVALID_WORK_DIMENSION,
  PI_INVALID_KERNEL_ARGS = CL_INVALID_KERNEL_ARGS,
  PI_INVALID_IMAGE_SIZE = CL_INVALID_IMAGE_SIZE,
  PI_INVALID_ARG_VALUE = CL_INVALID_ARG_VALUE,
  PI_INVALID_IMAGE_FORMAT_DESCRIPTOR = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
  PI_IMAGE_FORMAT_NOT_SUPPORTED = CL_IMAGE_FORMAT_NOT_SUPPORTED,
  PI_MEM_OBJECT_ALLOCATION_FAILURE = CL_MEM_OBJECT_ALLOCATION_FAILURE,
  PI_LINK_PROGRAM_FAILURE = CL_LINK_PROGRAM_FAILURE,
  PI_COMMAND_EXECUTION_FAILURE =
      -997, ///< PI_COMMAND_EXECUTION_FAILURE indicates an error occurred
            ///< during command enqueue or execution.
  PI_FUNCTION_ADDRESS_IS_NOT_AVAILABLE =
      -998, ///< PI_FUNCTION_ADDRESS_IS_NOT_AVAILABLE indicates a fallback
            ///< method determines the function exists but its address cannot be
            ///< found.
  PI_ERROR_UNKNOWN = -999
} _pi_result;

typedef enum {
  PI_EVENT_COMPLETE = CL_COMPLETE,
  PI_EVENT_RUNNING = CL_RUNNING,
  PI_EVENT_SUBMITTED = CL_SUBMITTED,
  PI_EVENT_QUEUED = CL_QUEUED
} _pi_event_status;

typedef enum {
  PI_PLATFORM_INFO_EXTENSIONS = CL_PLATFORM_EXTENSIONS,
  PI_PLATFORM_INFO_NAME = CL_PLATFORM_NAME,
  PI_PLATFORM_INFO_PROFILE = CL_PLATFORM_PROFILE,
  PI_PLATFORM_INFO_VENDOR = CL_PLATFORM_VENDOR,
  PI_PLATFORM_INFO_VERSION = CL_PLATFORM_VERSION
} _pi_platform_info;

typedef enum {
  PI_PROGRAM_BUILD_INFO_STATUS = CL_PROGRAM_BUILD_STATUS,
  PI_PROGRAM_BUILD_INFO_OPTIONS = CL_PROGRAM_BUILD_OPTIONS,
  PI_PROGRAM_BUILD_INFO_LOG = CL_PROGRAM_BUILD_LOG,
  PI_PROGRAM_BUILD_INFO_BINARY_TYPE = CL_PROGRAM_BINARY_TYPE
} _pi_program_build_info;

typedef enum {
  PI_PROGRAM_BUILD_STATUS_NONE = CL_BUILD_NONE,
  PI_PROGRAM_BUILD_STATUS_ERROR = CL_BUILD_ERROR,
  PI_PROGRAM_BUILD_STATUS_SUCCESS = CL_BUILD_SUCCESS,
  PI_PROGRAM_BUILD_STATUS_IN_PROGRESS = CL_BUILD_IN_PROGRESS
} _pi_program_build_status;

typedef enum {
  PI_PROGRAM_BINARY_TYPE_NONE = CL_PROGRAM_BINARY_TYPE_NONE,
  PI_PROGRAM_BINARY_TYPE_COMPILED_OBJECT =
      CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT,
  PI_PROGRAM_BINARY_TYPE_LIBRARY = CL_PROGRAM_BINARY_TYPE_LIBRARY,
  PI_PROGRAM_BINARY_TYPE_EXECUTABLE = CL_PROGRAM_BINARY_TYPE_EXECUTABLE
} _pi_program_binary_type;

// NOTE: this is made 64-bit to match the size of cl_device_type to
// make the translation to OpenCL transparent.
//
typedef enum : pi_uint64 {
  PI_DEVICE_TYPE_DEFAULT =
      CL_DEVICE_TYPE_DEFAULT, ///< The default device available in the PI
                              ///< plugin.
  PI_DEVICE_TYPE_ALL =
      CL_DEVICE_TYPE_ALL, ///< All devices available in the PI plugin.
  PI_DEVICE_TYPE_CPU =
      CL_DEVICE_TYPE_CPU, ///< A PI device that is the host processor.
  PI_DEVICE_TYPE_GPU = CL_DEVICE_TYPE_GPU, ///< A PI device that is a GPU.
  PI_DEVICE_TYPE_ACC = CL_DEVICE_TYPE_ACCELERATOR ///< A PI device that is a
                                                  ///< dedicated accelerator.
} _pi_device_type;

typedef enum {
  PI_DEVICE_MEM_CACHE_TYPE_NONE = CL_NONE,
  PI_DEVICE_MEM_CACHE_TYPE_READ_ONLY_CACHE = CL_READ_ONLY_CACHE,
  PI_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE = CL_READ_WRITE_CACHE
} _pi_device_mem_cache_type;

typedef enum {
  PI_DEVICE_LOCAL_MEM_TYPE_LOCAL = CL_LOCAL,
  PI_DEVICE_LOCAL_MEM_TYPE_GLOBAL = CL_GLOBAL
} _pi_device_local_mem_type;

typedef enum {
  PI_DEVICE_INFO_TYPE = CL_DEVICE_TYPE,
  PI_DEVICE_INFO_VENDOR_ID = CL_DEVICE_VENDOR_ID,
  PI_DEVICE_INFO_MAX_COMPUTE_UNITS = CL_DEVICE_MAX_COMPUTE_UNITS,
  PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
  PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES = CL_DEVICE_MAX_WORK_ITEM_SIZES,
  PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE = CL_DEVICE_MAX_WORK_GROUP_SIZE,
  PI_DEVICE_INFO_SINGLE_FP_CONFIG = CL_DEVICE_SINGLE_FP_CONFIG,
  PI_DEVICE_INFO_HALF_FP_CONFIG = CL_DEVICE_HALF_FP_CONFIG,
  PI_DEVICE_INFO_DOUBLE_FP_CONFIG = CL_DEVICE_DOUBLE_FP_CONFIG,
  PI_DEVICE_INFO_QUEUE_PROPERTIES = CL_DEVICE_QUEUE_PROPERTIES,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
  PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF =
      CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR = CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT =
      CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT = CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG = CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT =
      CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE =
      CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
  PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF = CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
  PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY = CL_DEVICE_MAX_CLOCK_FREQUENCY,
  PI_DEVICE_INFO_ADDRESS_BITS = CL_DEVICE_ADDRESS_BITS,
  PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE = CL_DEVICE_MAX_MEM_ALLOC_SIZE,
  PI_DEVICE_INFO_IMAGE_SUPPORT = CL_DEVICE_IMAGE_SUPPORT,
  PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS = CL_DEVICE_MAX_READ_IMAGE_ARGS,
  PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS = CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
  PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH = CL_DEVICE_IMAGE2D_MAX_WIDTH,
  PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT = CL_DEVICE_IMAGE2D_MAX_HEIGHT,
  PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH = CL_DEVICE_IMAGE3D_MAX_WIDTH,
  PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT = CL_DEVICE_IMAGE3D_MAX_HEIGHT,
  PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH = CL_DEVICE_IMAGE3D_MAX_DEPTH,
  PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE = CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
  PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE = CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
  PI_DEVICE_INFO_MAX_SAMPLERS = CL_DEVICE_MAX_SAMPLERS,
  PI_DEVICE_INFO_MAX_PARAMETER_SIZE = CL_DEVICE_MAX_PARAMETER_SIZE,
  PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN = CL_DEVICE_MEM_BASE_ADDR_ALIGN,
  PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
  PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE =
      CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
  PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
  PI_DEVICE_INFO_GLOBAL_MEM_SIZE = CL_DEVICE_GLOBAL_MEM_SIZE,
  PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
  PI_DEVICE_INFO_MAX_CONSTANT_ARGS = CL_DEVICE_MAX_CONSTANT_ARGS,
  PI_DEVICE_INFO_LOCAL_MEM_TYPE = CL_DEVICE_LOCAL_MEM_TYPE,
  PI_DEVICE_INFO_LOCAL_MEM_SIZE = CL_DEVICE_LOCAL_MEM_SIZE,
  PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT = CL_DEVICE_ERROR_CORRECTION_SUPPORT,
  PI_DEVICE_INFO_HOST_UNIFIED_MEMORY = CL_DEVICE_HOST_UNIFIED_MEMORY,
  PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION =
      CL_DEVICE_PROFILING_TIMER_RESOLUTION,
  PI_DEVICE_INFO_ENDIAN_LITTLE = CL_DEVICE_ENDIAN_LITTLE,
  PI_DEVICE_INFO_AVAILABLE = CL_DEVICE_AVAILABLE,
  PI_DEVICE_INFO_COMPILER_AVAILABLE = CL_DEVICE_COMPILER_AVAILABLE,
  PI_DEVICE_INFO_LINKER_AVAILABLE = CL_DEVICE_LINKER_AVAILABLE,
  PI_DEVICE_INFO_EXECUTION_CAPABILITIES = CL_DEVICE_EXECUTION_CAPABILITIES,
  PI_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES =
      CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
  PI_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES = CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
  PI_DEVICE_INFO_BUILT_IN_KERNELS = CL_DEVICE_BUILT_IN_KERNELS,
  PI_DEVICE_INFO_PLATFORM = CL_DEVICE_PLATFORM,
  PI_DEVICE_INFO_REFERENCE_COUNT = CL_DEVICE_REFERENCE_COUNT,
  PI_DEVICE_INFO_IL_VERSION = CL_DEVICE_IL_VERSION_KHR,
  PI_DEVICE_INFO_NAME = CL_DEVICE_NAME,
  PI_DEVICE_INFO_VENDOR = CL_DEVICE_VENDOR,
  PI_DEVICE_INFO_DRIVER_VERSION = CL_DRIVER_VERSION,
  PI_DEVICE_INFO_PROFILE = CL_DEVICE_PROFILE,
  PI_DEVICE_INFO_VERSION = CL_DEVICE_VERSION,
  PI_DEVICE_INFO_OPENCL_C_VERSION = CL_DEVICE_OPENCL_C_VERSION,
  PI_DEVICE_INFO_EXTENSIONS = CL_DEVICE_EXTENSIONS,
  PI_DEVICE_INFO_PRINTF_BUFFER_SIZE = CL_DEVICE_PRINTF_BUFFER_SIZE,
  PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC =
      CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
  PI_DEVICE_INFO_PARENT_DEVICE = CL_DEVICE_PARENT_DEVICE,
  PI_DEVICE_INFO_PARTITION_PROPERTIES = CL_DEVICE_PARTITION_PROPERTIES,
  PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES =
      CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
  PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN =
      CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
  PI_DEVICE_INFO_PARTITION_TYPE = CL_DEVICE_PARTITION_TYPE,
  PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS = CL_DEVICE_MAX_NUM_SUB_GROUPS,
  PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS =
      CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
  PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL = CL_DEVICE_SUB_GROUP_SIZES_INTEL,
  PI_DEVICE_INFO_USM_HOST_SUPPORT = CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
  PI_DEVICE_INFO_USM_DEVICE_SUPPORT = CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
  PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT =
      CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
  PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT =
      CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
  PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT =
      CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
  // Intel UUID extension.
  PI_DEVICE_INFO_UUID = CL_DEVICE_UUID_KHR,
  // These are Intel-specific extensions.
  PI_DEVICE_INFO_PCI_ADDRESS = 0x10020,
  PI_DEVICE_INFO_GPU_EU_COUNT = 0x10021,
  PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH = 0x10022,
  PI_DEVICE_INFO_GPU_SLICES = 0x10023,
  PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE = 0x10024,
  PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE = 0x10025,
  PI_DEVICE_INFO_MAX_MEM_BANDWIDTH = 0x10026,
  PI_DEVICE_INFO_IMAGE_SRGB = 0x10027,
  PI_DEVICE_INFO_HOMOGENEOUS_ARCH = 0x10028,
  PI_DEVICE_INFO_ATOMIC_64 = 0x10110,
  PI_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES = 0x10111,
  PI_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES = 0x11000,
  PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU = 0x10112,
  PI_EXT_ONEAPI_DEVICE_INFO_MAX_GLOBAL_WORK_GROUPS = 0x20000,
  PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_1D = 0x20001,
  PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_2D = 0x20002,
  PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D = 0x20003
} _pi_device_info;

typedef enum {
  PI_PROGRAM_INFO_REFERENCE_COUNT = CL_PROGRAM_REFERENCE_COUNT,
  PI_PROGRAM_INFO_CONTEXT = CL_PROGRAM_CONTEXT,
  PI_PROGRAM_INFO_NUM_DEVICES = CL_PROGRAM_NUM_DEVICES,
  PI_PROGRAM_INFO_DEVICES = CL_PROGRAM_DEVICES,
  PI_PROGRAM_INFO_SOURCE = CL_PROGRAM_SOURCE,
  PI_PROGRAM_INFO_BINARY_SIZES = CL_PROGRAM_BINARY_SIZES,
  PI_PROGRAM_INFO_BINARIES = CL_PROGRAM_BINARIES,
  PI_PROGRAM_INFO_NUM_KERNELS = CL_PROGRAM_NUM_KERNELS,
  PI_PROGRAM_INFO_KERNEL_NAMES = CL_PROGRAM_KERNEL_NAMES
} _pi_program_info;

typedef enum {
  PI_CONTEXT_INFO_DEVICES = CL_CONTEXT_DEVICES,
  PI_CONTEXT_INFO_NUM_DEVICES = CL_CONTEXT_NUM_DEVICES,
  PI_CONTEXT_INFO_PROPERTIES = CL_CONTEXT_PROPERTIES,
  PI_CONTEXT_INFO_REFERENCE_COUNT = CL_CONTEXT_REFERENCE_COUNT,
  // Atomics capabilities extensions
  PI_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES = 0x10010,
  PI_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES = 0x10011
} _pi_context_info;

typedef enum {
  PI_QUEUE_INFO_CONTEXT = CL_QUEUE_CONTEXT,
  PI_QUEUE_INFO_DEVICE = CL_QUEUE_DEVICE,
  PI_QUEUE_INFO_DEVICE_DEFAULT = CL_QUEUE_DEVICE_DEFAULT,
  PI_QUEUE_INFO_PROPERTIES = CL_QUEUE_PROPERTIES,
  PI_QUEUE_INFO_REFERENCE_COUNT = CL_QUEUE_REFERENCE_COUNT,
  PI_QUEUE_INFO_SIZE = CL_QUEUE_SIZE
} _pi_queue_info;

typedef enum {
  PI_KERNEL_INFO_FUNCTION_NAME = CL_KERNEL_FUNCTION_NAME,
  PI_KERNEL_INFO_NUM_ARGS = CL_KERNEL_NUM_ARGS,
  PI_KERNEL_INFO_REFERENCE_COUNT = CL_KERNEL_REFERENCE_COUNT,
  PI_KERNEL_INFO_CONTEXT = CL_KERNEL_CONTEXT,
  PI_KERNEL_INFO_PROGRAM = CL_KERNEL_PROGRAM,
  PI_KERNEL_INFO_ATTRIBUTES = CL_KERNEL_ATTRIBUTES
} _pi_kernel_info;

typedef enum {
  PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE = CL_KERNEL_GLOBAL_WORK_SIZE,
  PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE = CL_KERNEL_WORK_GROUP_SIZE,
  PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE =
      CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
  PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE = CL_KERNEL_LOCAL_MEM_SIZE,
  PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE =
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
  PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE = CL_KERNEL_PRIVATE_MEM_SIZE,
  // The number of registers used by the compiled kernel (device specific)
  PI_KERNEL_GROUP_INFO_NUM_REGS = 0x10112
} _pi_kernel_group_info;

typedef enum {
  PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
  PI_FP_ROUND_TO_NEAREST = CL_FP_ROUND_TO_NEAREST,
  PI_FP_ROUND_TO_ZERO = CL_FP_ROUND_TO_ZERO,
  PI_FP_ROUND_TO_INF = CL_FP_ROUND_TO_INF,
  PI_FP_INF_NAN = CL_FP_INF_NAN,
  PI_FP_DENORM = CL_FP_DENORM,
  PI_FP_FMA = CL_FP_FMA
} _pi_fp_capabilities;

typedef enum {
  PI_IMAGE_INFO_FORMAT = CL_IMAGE_FORMAT,
  PI_IMAGE_INFO_ELEMENT_SIZE = CL_IMAGE_ELEMENT_SIZE,
  PI_IMAGE_INFO_ROW_PITCH = CL_IMAGE_ROW_PITCH,
  PI_IMAGE_INFO_SLICE_PITCH = CL_IMAGE_SLICE_PITCH,
  PI_IMAGE_INFO_WIDTH = CL_IMAGE_WIDTH,
  PI_IMAGE_INFO_HEIGHT = CL_IMAGE_HEIGHT,
  PI_IMAGE_INFO_DEPTH = CL_IMAGE_DEPTH
} _pi_image_info;

typedef enum {
  PI_KERNEL_MAX_SUB_GROUP_SIZE = CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
  PI_KERNEL_MAX_NUM_SUB_GROUPS = CL_KERNEL_MAX_NUM_SUB_GROUPS,
  PI_KERNEL_COMPILE_NUM_SUB_GROUPS = CL_KERNEL_COMPILE_NUM_SUB_GROUPS,
  PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL =
      CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL
} _pi_kernel_sub_group_info;

typedef enum {
  PI_EVENT_INFO_COMMAND_QUEUE = CL_EVENT_COMMAND_QUEUE,
  PI_EVENT_INFO_CONTEXT = CL_EVENT_CONTEXT,
  PI_EVENT_INFO_COMMAND_TYPE = CL_EVENT_COMMAND_TYPE,
  PI_EVENT_INFO_COMMAND_EXECUTION_STATUS = CL_EVENT_COMMAND_EXECUTION_STATUS,
  PI_EVENT_INFO_REFERENCE_COUNT = CL_EVENT_REFERENCE_COUNT
} _pi_event_info;

typedef enum {
  PI_COMMAND_TYPE_NDRANGE_KERNEL = CL_COMMAND_NDRANGE_KERNEL,
  PI_COMMAND_TYPE_MEM_BUFFER_READ = CL_COMMAND_READ_BUFFER,
  PI_COMMAND_TYPE_MEM_BUFFER_WRITE = CL_COMMAND_WRITE_BUFFER,
  PI_COMMAND_TYPE_MEM_BUFFER_COPY = CL_COMMAND_COPY_BUFFER,
  PI_COMMAND_TYPE_MEM_BUFFER_MAP = CL_COMMAND_MAP_BUFFER,
  PI_COMMAND_TYPE_MEM_BUFFER_UNMAP = CL_COMMAND_UNMAP_MEM_OBJECT,
  PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT = CL_COMMAND_READ_BUFFER_RECT,
  PI_COMMAND_TYPE_MEM_BUFFER_WRITE_RECT = CL_COMMAND_WRITE_BUFFER_RECT,
  PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT = CL_COMMAND_COPY_BUFFER_RECT,
  PI_COMMAND_TYPE_USER = CL_COMMAND_USER,
  PI_COMMAND_TYPE_MEM_BUFFER_FILL = CL_COMMAND_FILL_BUFFER,
  PI_COMMAND_TYPE_IMAGE_READ = CL_COMMAND_READ_IMAGE,
  PI_COMMAND_TYPE_IMAGE_WRITE = CL_COMMAND_WRITE_IMAGE,
  PI_COMMAND_TYPE_IMAGE_COPY = CL_COMMAND_COPY_IMAGE,
  PI_COMMAND_TYPE_NATIVE_KERNEL = CL_COMMAND_NATIVE_KERNEL,
  PI_COMMAND_TYPE_COPY_BUFFER_TO_IMAGE = CL_COMMAND_COPY_BUFFER_TO_IMAGE,
  PI_COMMAND_TYPE_COPY_IMAGE_TO_BUFFER = CL_COMMAND_COPY_IMAGE_TO_BUFFER,
  PI_COMMAND_TYPE_MAP_IMAGE = CL_COMMAND_MAP_IMAGE,
  PI_COMMAND_TYPE_MARKER = CL_COMMAND_MARKER,
  PI_COMMAND_TYPE_ACQUIRE_GL_OBJECTS = CL_COMMAND_ACQUIRE_GL_OBJECTS,
  PI_COMMAND_TYPE_RELEASE_GL_OBJECTS = CL_COMMAND_RELEASE_GL_OBJECTS,
  PI_COMMAND_TYPE_BARRIER = CL_COMMAND_BARRIER,
  PI_COMMAND_TYPE_MIGRATE_MEM_OBJECTS = CL_COMMAND_MIGRATE_MEM_OBJECTS,
  PI_COMMAND_TYPE_FILL_IMAGE = CL_COMMAND_FILL_IMAGE,
  PI_COMMAND_TYPE_SVM_FREE = CL_COMMAND_SVM_FREE,
  PI_COMMAND_TYPE_SVM_MEMCPY = CL_COMMAND_SVM_MEMCPY,
  PI_COMMAND_TYPE_SVM_MEMFILL = CL_COMMAND_SVM_MEMFILL,
  PI_COMMAND_TYPE_SVM_MAP = CL_COMMAND_SVM_MAP,
  PI_COMMAND_TYPE_SVM_UNMAP = CL_COMMAND_SVM_UNMAP
} _pi_command_type;

typedef enum {
  PI_MEM_TYPE_BUFFER = CL_MEM_OBJECT_BUFFER,
  PI_MEM_TYPE_IMAGE2D = CL_MEM_OBJECT_IMAGE2D,
  PI_MEM_TYPE_IMAGE3D = CL_MEM_OBJECT_IMAGE3D,
  PI_MEM_TYPE_IMAGE2D_ARRAY = CL_MEM_OBJECT_IMAGE2D_ARRAY,
  PI_MEM_TYPE_IMAGE1D = CL_MEM_OBJECT_IMAGE1D,
  PI_MEM_TYPE_IMAGE1D_ARRAY = CL_MEM_OBJECT_IMAGE1D_ARRAY,
  PI_MEM_TYPE_IMAGE1D_BUFFER = CL_MEM_OBJECT_IMAGE1D_BUFFER
} _pi_mem_type;

typedef enum {
  // Device-specific value opaque in PI API.
  PI_MEM_ADVICE_UNKNOWN,
  PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY = 101,
  PI_MEM_ADVICE_CUDA_UNSET_READ_MOSTLY = 102,
  PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION = 103,
  PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION = 104,
  PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY = 105,
  PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY = 106,
  PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION_HOST = 107,
  PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION_HOST = 108,
  PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY_HOST = 109,
  PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY_HOST = 110,
} _pi_mem_advice;

typedef enum {
  PI_IMAGE_CHANNEL_ORDER_A = CL_A,
  PI_IMAGE_CHANNEL_ORDER_R = CL_R,
  PI_IMAGE_CHANNEL_ORDER_RG = CL_RG,
  PI_IMAGE_CHANNEL_ORDER_RA = CL_RA,
  PI_IMAGE_CHANNEL_ORDER_RGB = CL_RGB,
  PI_IMAGE_CHANNEL_ORDER_RGBA = CL_RGBA,
  PI_IMAGE_CHANNEL_ORDER_BGRA = CL_BGRA,
  PI_IMAGE_CHANNEL_ORDER_ARGB = CL_ARGB,
  PI_IMAGE_CHANNEL_ORDER_ABGR = CL_ABGR,
  PI_IMAGE_CHANNEL_ORDER_INTENSITY = CL_INTENSITY,
  PI_IMAGE_CHANNEL_ORDER_LUMINANCE = CL_LUMINANCE,
  PI_IMAGE_CHANNEL_ORDER_Rx = CL_Rx,
  PI_IMAGE_CHANNEL_ORDER_RGx = CL_RGx,
  PI_IMAGE_CHANNEL_ORDER_RGBx = CL_RGBx,
  PI_IMAGE_CHANNEL_ORDER_sRGBA = CL_sRGBA
} _pi_image_channel_order;

typedef enum {
  PI_IMAGE_CHANNEL_TYPE_SNORM_INT8 = CL_SNORM_INT8,
  PI_IMAGE_CHANNEL_TYPE_SNORM_INT16 = CL_SNORM_INT16,
  PI_IMAGE_CHANNEL_TYPE_UNORM_INT8 = CL_UNORM_INT8,
  PI_IMAGE_CHANNEL_TYPE_UNORM_INT16 = CL_UNORM_INT16,
  PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = CL_UNORM_SHORT_565,
  PI_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = CL_UNORM_SHORT_555,
  PI_IMAGE_CHANNEL_TYPE_UNORM_INT_101010 = CL_UNORM_INT_101010,
  PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = CL_SIGNED_INT8,
  PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = CL_SIGNED_INT16,
  PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = CL_SIGNED_INT32,
  PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = CL_UNSIGNED_INT8,
  PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = CL_UNSIGNED_INT16,
  PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = CL_UNSIGNED_INT32,
  PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT = CL_HALF_FLOAT,
  PI_IMAGE_CHANNEL_TYPE_FLOAT = CL_FLOAT
} _pi_image_channel_type;

typedef enum {
  PI_BUFFER_CREATE_TYPE_REGION = CL_BUFFER_CREATE_TYPE_REGION
} _pi_buffer_create_type;

const pi_bool PI_TRUE = CL_TRUE;
const pi_bool PI_FALSE = CL_FALSE;

typedef enum {
  PI_SAMPLER_INFO_REFERENCE_COUNT = CL_SAMPLER_REFERENCE_COUNT,
  PI_SAMPLER_INFO_CONTEXT = CL_SAMPLER_CONTEXT,
  PI_SAMPLER_INFO_NORMALIZED_COORDS = CL_SAMPLER_NORMALIZED_COORDS,
  PI_SAMPLER_INFO_ADDRESSING_MODE = CL_SAMPLER_ADDRESSING_MODE,
  PI_SAMPLER_INFO_FILTER_MODE = CL_SAMPLER_FILTER_MODE,
  PI_SAMPLER_INFO_MIP_FILTER_MODE = CL_SAMPLER_MIP_FILTER_MODE,
  PI_SAMPLER_INFO_LOD_MIN = CL_SAMPLER_LOD_MIN,
  PI_SAMPLER_INFO_LOD_MAX = CL_SAMPLER_LOD_MAX
} _pi_sampler_info;

typedef enum {
  PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = CL_ADDRESS_MIRRORED_REPEAT,
  PI_SAMPLER_ADDRESSING_MODE_REPEAT = CL_ADDRESS_REPEAT,
  PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = CL_ADDRESS_CLAMP_TO_EDGE,
  PI_SAMPLER_ADDRESSING_MODE_CLAMP = CL_ADDRESS_CLAMP,
  PI_SAMPLER_ADDRESSING_MODE_NONE = CL_ADDRESS_NONE
} _pi_sampler_addressing_mode;

typedef enum {
  PI_SAMPLER_FILTER_MODE_NEAREST = CL_FILTER_NEAREST,
  PI_SAMPLER_FILTER_MODE_LINEAR = CL_FILTER_LINEAR,
} _pi_sampler_filter_mode;

using pi_context_properties = intptr_t;

using pi_device_exec_capabilities = pi_bitfield;
constexpr pi_device_exec_capabilities PI_DEVICE_EXEC_CAPABILITIES_KERNEL =
    CL_EXEC_KERNEL;
constexpr pi_device_exec_capabilities
    PI_DEVICE_EXEC_CAPABILITIES_NATIVE_KERNEL = CL_EXEC_NATIVE_KERNEL;

using pi_sampler_properties = pi_bitfield;
constexpr pi_sampler_properties PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS =
    CL_SAMPLER_NORMALIZED_COORDS;
constexpr pi_sampler_properties PI_SAMPLER_PROPERTIES_ADDRESSING_MODE =
    CL_SAMPLER_ADDRESSING_MODE;
constexpr pi_sampler_properties PI_SAMPLER_PROPERTIES_FILTER_MODE =
    CL_SAMPLER_FILTER_MODE;

using pi_memory_order_capabilities = pi_bitfield;
constexpr pi_memory_order_capabilities PI_MEMORY_ORDER_RELAXED = 0x01;
constexpr pi_memory_order_capabilities PI_MEMORY_ORDER_ACQUIRE = 0x02;
constexpr pi_memory_order_capabilities PI_MEMORY_ORDER_RELEASE = 0x04;
constexpr pi_memory_order_capabilities PI_MEMORY_ORDER_ACQ_REL = 0x08;
constexpr pi_memory_order_capabilities PI_MEMORY_ORDER_SEQ_CST = 0x10;

using pi_memory_scope_capabilities = pi_bitfield;
constexpr pi_memory_scope_capabilities PI_MEMORY_SCOPE_WORK_ITEM = 0x01;
constexpr pi_memory_scope_capabilities PI_MEMORY_SCOPE_SUB_GROUP = 0x02;
constexpr pi_memory_scope_capabilities PI_MEMORY_SCOPE_WORK_GROUP = 0x04;
constexpr pi_memory_scope_capabilities PI_MEMORY_SCOPE_DEVICE = 0x08;
constexpr pi_memory_scope_capabilities PI_MEMORY_SCOPE_SYSTEM = 0x10;

typedef enum {
  PI_PROFILING_INFO_COMMAND_QUEUED = CL_PROFILING_COMMAND_QUEUED,
  PI_PROFILING_INFO_COMMAND_SUBMIT = CL_PROFILING_COMMAND_SUBMIT,
  PI_PROFILING_INFO_COMMAND_START = CL_PROFILING_COMMAND_START,
  PI_PROFILING_INFO_COMMAND_END = CL_PROFILING_COMMAND_END
} _pi_profiling_info;

// NOTE: this is made 64-bit to match the size of cl_mem_flags to
// make the translation to OpenCL transparent.
// TODO: populate
//
using pi_mem_flags = pi_bitfield;
// Access
constexpr pi_mem_flags PI_MEM_FLAGS_ACCESS_RW = CL_MEM_READ_WRITE;
constexpr pi_mem_flags PI_MEM_ACCESS_READ_ONLY = CL_MEM_READ_ONLY;
// Host pointer
constexpr pi_mem_flags PI_MEM_FLAGS_HOST_PTR_USE = CL_MEM_USE_HOST_PTR;
constexpr pi_mem_flags PI_MEM_FLAGS_HOST_PTR_COPY = CL_MEM_COPY_HOST_PTR;
constexpr pi_mem_flags PI_MEM_FLAGS_HOST_PTR_ALLOC = CL_MEM_ALLOC_HOST_PTR;

// flags passed to Map operations
using pi_map_flags = pi_bitfield;
constexpr pi_map_flags PI_MAP_READ = CL_MAP_READ;
constexpr pi_map_flags PI_MAP_WRITE = CL_MAP_WRITE;
constexpr pi_map_flags PI_MAP_WRITE_INVALIDATE_REGION =
    CL_MAP_WRITE_INVALIDATE_REGION;

// NOTE: this is made 64-bit to match the size of cl_mem_properties_intel to
// make the translation to OpenCL transparent.
using pi_mem_properties = pi_bitfield;
constexpr pi_mem_properties PI_MEM_PROPERTIES_CHANNEL = CL_MEM_CHANNEL_INTEL;

// NOTE: queue properties are implemented this way to better support bit
// manipulations
using pi_queue_properties = pi_bitfield;
constexpr pi_queue_properties PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE =
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
constexpr pi_queue_properties PI_QUEUE_PROFILING_ENABLE =
    CL_QUEUE_PROFILING_ENABLE;
constexpr pi_queue_properties PI_QUEUE_ON_DEVICE = CL_QUEUE_ON_DEVICE;
constexpr pi_queue_properties PI_QUEUE_ON_DEVICE_DEFAULT =
    CL_QUEUE_ON_DEVICE_DEFAULT;

using pi_result = _pi_result;
using pi_platform_info = _pi_platform_info;
using pi_device_type = _pi_device_type;
using pi_device_mem_cache_type = _pi_device_mem_cache_type;
using pi_device_local_mem_type = _pi_device_local_mem_type;
using pi_device_info = _pi_device_info;
using pi_program_info = _pi_program_info;
using pi_context_info = _pi_context_info;
using pi_queue_info = _pi_queue_info;
using pi_image_info = _pi_image_info;
using pi_kernel_info = _pi_kernel_info;
using pi_kernel_group_info = _pi_kernel_group_info;
using pi_kernel_sub_group_info = _pi_kernel_sub_group_info;
using pi_fp_capabilities = _pi_fp_capabilities;
using pi_event_info = _pi_event_info;
using pi_command_type = _pi_command_type;
using pi_mem_type = _pi_mem_type;
using pi_mem_advice = _pi_mem_advice;
using pi_image_channel_order = _pi_image_channel_order;
using pi_image_channel_type = _pi_image_channel_type;
using pi_buffer_create_type = _pi_buffer_create_type;
using pi_sampler_addressing_mode = _pi_sampler_addressing_mode;
using pi_sampler_filter_mode = _pi_sampler_filter_mode;
using pi_sampler_info = _pi_sampler_info;
using pi_event_status = _pi_event_status;
using pi_program_build_info = _pi_program_build_info;
using pi_program_build_status = _pi_program_build_status;
using pi_kernel_info = _pi_kernel_info;
using pi_profiling_info = _pi_profiling_info;

// For compatibility with OpenCL define this not as enum.
using pi_device_partition_property = intptr_t;
static constexpr pi_device_partition_property PI_DEVICE_PARTITION_EQUALLY =
    CL_DEVICE_PARTITION_EQUALLY;
static constexpr pi_device_partition_property
    PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN =
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;

// For compatibility with OpenCL define this not as enum.
using pi_device_affinity_domain = pi_bitfield;
static constexpr pi_device_affinity_domain PI_DEVICE_AFFINITY_DOMAIN_NUMA =
    CL_DEVICE_AFFINITY_DOMAIN_NUMA;
static constexpr pi_device_affinity_domain
    PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE =
        CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;

// Entry type, matches OpenMP for compatibility
struct _pi_offload_entry_struct {
  void *addr;
  char *name;
  size_t size;
  int32_t flags;
  int32_t reserved;
};

using _pi_offload_entry = _pi_offload_entry_struct *;

// A type of a binary image property.
typedef enum {
  PI_PROPERTY_TYPE_UNKNOWN,
  PI_PROPERTY_TYPE_UINT32,     // 32-bit integer
  PI_PROPERTY_TYPE_BYTE_ARRAY, // byte array
  PI_PROPERTY_TYPE_STRING      // null-terminated string
} pi_property_type;

// Device binary image property.
// If the type size of the property value is fixed and is no greater than
// 64 bits, then ValAddr is 0 and the value is stored in the ValSize field.
// Example - PI_PROPERTY_TYPE_UINT32, which is 32-bit
struct _pi_device_binary_property_struct {
  char *Name;       // null-terminated property name
  void *ValAddr;    // address of property value
  uint32_t Type;    // _pi_property_type
  uint64_t ValSize; // size of property value in bytes
};

typedef _pi_device_binary_property_struct *pi_device_binary_property;

// Named array of properties.
struct _pi_device_binary_property_set_struct {
  char *Name;                                // the name
  pi_device_binary_property PropertiesBegin; // array start
  pi_device_binary_property PropertiesEnd;   // array end
};

typedef _pi_device_binary_property_set_struct *pi_device_binary_property_set;

/// Types of device binary.
using pi_device_binary_type = uint8_t;
// format is not determined
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_NONE = 0;
// specific to a device
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_NATIVE = 1;
// portable binary types go next
// SPIR-V
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_SPIRV = 2;
// LLVM bitcode
static constexpr pi_device_binary_type PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE = 3;

// Device binary descriptor version supported by this library.
static const uint16_t PI_DEVICE_BINARY_VERSION = 1;

// The kind of offload model the binary employs; must be 4 for SYCL
static const uint8_t PI_DEVICE_BINARY_OFFLOAD_KIND_SYCL = 4;

/// Target identification strings for
/// pi_device_binary_struct.DeviceTargetSpec
///
/// A device type represented by a particular target
/// triple requires specific binary images. We need
/// to map the image type onto the device target triple
///
#define __SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN "<unknown>"
/// SPIR-V 32-bit image <-> "spir", 32-bit OpenCL device
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV32 "spir"
/// SPIR-V 64-bit image <-> "spir64", 64-bit OpenCL device
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64 "spir64"
/// Device-specific binary images produced from SPIR-V 64-bit <->
/// various "spir64_*" triples for specific 64-bit OpenCL devices
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64 "spir64_x86_64"
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN "spir64_gen"
#define __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA "spir64_fpga"
/// PTX 64-bit image <-> "nvptx64", 64-bit NVIDIA PTX device
#define __SYCL_PI_DEVICE_BINARY_TARGET_NVPTX64 "nvptx64"
#define __SYCL_PI_DEVICE_BINARY_TARGET_AMDGCN "amdgcn"

/// Extension to denote native support of assert feature by an arbitrary device
/// piDeviceGetInfo call should return this extension when the device supports
/// native asserts if supported extensions' names are requested
#define PI_DEVICE_INFO_EXTENSION_DEVICELIB_ASSERT                              \
  "pi_ext_intel_devicelib_assert"

/// Device binary image property set names recognized by the SYCL runtime.
/// Name must be consistent with
/// PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS defined in
/// PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SPEC_CONST_MAP "SYCL/specialization constants"
/// PropertySetRegistry::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES defined in
/// PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SPEC_CONST_DEFAULT_VALUES_MAP                   \
  "SYCL/specialization constants default values"
/// PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_DEVICELIB_REQ_MASK "SYCL/devicelib req mask"
/// PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO "SYCL/kernel param opt"
/// PropertySetRegistry::SYCL_KERNEL_PROGRAM_METADATA defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_PROGRAM_METADATA "SYCL/program metadata"
/// PropertySetRegistry::SYCL_MISC_PROP defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_MISC_PROP "SYCL/misc properties"
/// PropertySetRegistry::SYCL_ASSERT_USED defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_ASSERT_USED "SYCL/assert used"
/// PropertySetRegistry::SYCL_EXPORTED_SYMBOLS defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS "SYCL/exported symbols"
/// PropertySetRegistry::SYCL_DEVICE_GLOBALS defined in PropertySetIO.h
#define __SYCL_PI_PROPERTY_SET_SYCL_DEVICE_GLOBALS "SYCL/device globals"

/// Program metadata tags recognized by the PI backends. For kernels the tag
/// must appear after the kernel name.
#define __SYCL_PI_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE                    \
  "@reqd_work_group_size"

/// This struct is a record of the device binary information. If the Kind field
/// denotes a portable binary type (SPIR-V or LLVM IR), the DeviceTargetSpec
/// field can still be specific and denote e.g. FPGA target. It must match the
/// __tgt_device_image structure generated by the clang-offload-wrapper tool
/// when their Version field match.
struct pi_device_binary_struct {
  /// version of this structure - for backward compatibility;
  /// all modifications which change order/type/offsets of existing fields
  /// should increment the version.
  uint16_t Version;
  /// the type of offload model the binary employs; must be 4 for SYCL
  uint8_t Kind;
  /// format of the binary data - SPIR-V, LLVM IR bitcode,...
  uint8_t Format;
  /// null-terminated string representation of the device's target architecture
  /// which holds one of:
  /// __SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN - unknown
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV32 - general value for 32-bit OpenCL
  /// devices
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64 - general value for 64-bit OpenCL
  /// devices
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64 - 64-bit OpenCL CPU device
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN - GEN GPU device (64-bit
  /// OpenCL)
  /// __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA - 64-bit OpenCL FPGA device
  const char *DeviceTargetSpec;
  /// a null-terminated string; target- and compiler-specific options
  /// which are suggested to use to "compile" program at runtime
  const char *CompileOptions;
  /// a null-terminated string; target- and compiler-specific options
  /// which are suggested to use to "link" program at runtime
  const char *LinkOptions;
  /// Pointer to the manifest data start
  const char *ManifestStart;
  /// Pointer to the manifest data end
  const char *ManifestEnd;
  /// Pointer to the target code start
  const unsigned char *BinaryStart;
  /// Pointer to the target code end
  const unsigned char *BinaryEnd;
  /// the offload entry table
  _pi_offload_entry EntriesBegin;
  _pi_offload_entry EntriesEnd;
  // Array of preperty sets; e.g. specialization constants symbol-int ID map is
  // propagated to runtime with this mechanism.
  pi_device_binary_property_set PropertySetsBegin;
  pi_device_binary_property_set PropertySetsEnd;
  // TODO Other fields like entries, link options can be propagated using
  // the property set infrastructure. This will improve binary compatibility and
  // add flexibility.
};
using pi_device_binary = pi_device_binary_struct *;

// pi_buffer_region structure repeats cl_buffer_region, used for sub buffers.
struct pi_buffer_region_struct {
  size_t origin;
  size_t size;
};
using pi_buffer_region = pi_buffer_region_struct *;

// pi_buff_rect_offset structure is 3D offset argument passed to buffer rect
// operations (piEnqueueMemBufferCopyRect, etc).
struct pi_buff_rect_offset_struct {
  size_t x_bytes;
  size_t y_scalar;
  size_t z_scalar;
};
using pi_buff_rect_offset = pi_buff_rect_offset_struct *;

// pi_buff_rect_region structure represents size of 3D region passed to buffer
// rect operations (piEnqueueMemBufferCopyRect, etc).
struct pi_buff_rect_region_struct {
  size_t width_bytes;
  size_t height_scalar;
  size_t depth_scalar;
};
using pi_buff_rect_region = pi_buff_rect_region_struct *;

// pi_image_offset structure is 3D offset argument passed to image operations
// (piEnqueueMemImageRead, etc).
struct pi_image_offset_struct {
  size_t x;
  size_t y;
  size_t z;
};
using pi_image_offset = pi_image_offset_struct *;

// pi_image_region structure represents size of 3D region passed to image
// operations (piEnqueueMemImageRead, etc).
struct pi_image_region_struct {
  size_t width;
  size_t height;
  size_t depth;
};
using pi_image_region = pi_image_region_struct *;

// Offload binaries descriptor version supported by this library.
static const uint16_t PI_DEVICE_BINARIES_VERSION = 1;

/// This struct is a record of all the device code that may be offloaded.
/// It must match the __tgt_bin_desc structure generated by
/// the clang-offload-wrapper tool when their Version field match.
struct pi_device_binaries_struct {
  /// version of this structure - for backward compatibility;
  /// all modifications which change order/type/offsets of existing fields
  /// should increment the version.
  uint16_t Version;
  /// Number of device binaries in this descriptor
  uint16_t NumDeviceBinaries;
  /// Device binaries data
  pi_device_binary DeviceBinaries;
  /// the offload entry table (not used, for compatibility with OpenMP)
  _pi_offload_entry *HostEntriesBegin;
  _pi_offload_entry *HostEntriesEnd;
};
using pi_device_binaries = pi_device_binaries_struct *;

// Opaque types that make reading build log errors easier.
struct _pi_platform;
struct _pi_device;
struct _pi_context;
struct _pi_queue;
struct _pi_mem;
struct _pi_program;
struct _pi_kernel;
struct _pi_event;
struct _pi_sampler;

using pi_platform = _pi_platform *;
using pi_device = _pi_device *;
using pi_context = _pi_context *;
using pi_queue = _pi_queue *;
using pi_mem = _pi_mem *;
using pi_program = _pi_program *;
using pi_kernel = _pi_kernel *;
using pi_event = _pi_event *;
using pi_sampler = _pi_sampler *;

typedef struct {
  pi_image_channel_order image_channel_order;
  pi_image_channel_type image_channel_data_type;
} _pi_image_format;

typedef struct {
  pi_mem_type image_type;
  size_t image_width;
  size_t image_height;
  size_t image_depth;
  size_t image_array_size;
  size_t image_row_pitch;
  size_t image_slice_pitch;
  pi_uint32 num_mip_levels;
  pi_uint32 num_samples;
  pi_mem buffer;
} _pi_image_desc;

using pi_image_format = _pi_image_format;
using pi_image_desc = _pi_image_desc;
//
// Following section contains SYCL RT Plugin Interface (PI) functions.
// They are 3 distinct categories:
//
// 1) Ones having direct analogy in OpenCL and needed for the core SYCL
//    functionality are started with just "pi" prefix in their names.
// 2) Those having direct analogy in OpenCL but only needed for SYCL
//    interoperability with OpenCL are started with "picl" prefix.
// 3) Functions having no direct analogy in OpenCL, started with "piext".
//
// TODO: describe interfaces in Doxygen format
//

struct _pi_plugin;
using pi_plugin = _pi_plugin;

// PI Plugin Initialise.
// Plugin will check the PI version of Plugin Interface,
// populate the PI Version it supports, update targets field and populate
// PiFunctionTable with Supported APIs. The pointers are in a predetermined
// order in pi.def file.
__SYCL_EXPORT pi_result piPluginInit(pi_plugin *plugin_info);

//
// Platform
//
__SYCL_EXPORT pi_result piPlatformsGet(pi_uint32 num_entries,
                                       pi_platform *platforms,
                                       pi_uint32 *num_platforms);

__SYCL_EXPORT pi_result piPlatformGetInfo(pi_platform platform,
                                          pi_platform_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret);

/// Gets the native handle of a PI platform object.
///
/// \param platform is the PI platform to get the native handle of.
/// \param nativeHandle is the native handle of platform.
__SYCL_EXPORT pi_result piextPlatformGetNativeHandle(
    pi_platform platform, pi_native_handle *nativeHandle);

/// Creates PI platform object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI device from.
/// \param platform is the PI platform created from the native handle.
__SYCL_EXPORT pi_result piextPlatformCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_platform *platform);

__SYCL_EXPORT pi_result piDevicesGet(pi_platform platform,
                                     pi_device_type device_type,
                                     pi_uint32 num_entries, pi_device *devices,
                                     pi_uint32 *num_devices);

/// Returns requested info for provided native device
/// Return PI_DEVICE_INFO_EXTENSION_DEVICELIB_ASSERT for
/// PI_DEVICE_INFO_EXTENSIONS query when the device supports native asserts
__SYCL_EXPORT pi_result piDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piDeviceRetain(pi_device device);

__SYCL_EXPORT pi_result piDeviceRelease(pi_device device);

__SYCL_EXPORT pi_result piDevicePartition(
    pi_device device, const pi_device_partition_property *properties,
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices);

/// Gets the native handle of a PI device object.
///
/// \param device is the PI device to get the native handle of.
/// \param nativeHandle is the native handle of device.
__SYCL_EXPORT pi_result
piextDeviceGetNativeHandle(pi_device device, pi_native_handle *nativeHandle);

/// Creates PI device object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI device from.
/// \param platform is the platform of the device (optional).
/// \param device is the PI device created from the native handle.
__SYCL_EXPORT pi_result piextDeviceCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_platform platform, pi_device *device);

/// Selects the most appropriate device binary based on runtime information
/// and the IR characteristics.
///
__SYCL_EXPORT pi_result piextDeviceSelectBinary(pi_device device,
                                                pi_device_binary *binaries,
                                                pi_uint32 num_binaries,
                                                pi_uint32 *selected_binary_ind);

/// Retrieves a device function pointer to a user-defined function
/// \arg \c function_name. \arg \c function_pointer_ret is set to 0 if query
/// failed.
///
/// \arg \c program must be built before calling this API. \arg \c device
/// must present in the list of devices returned by \c get_device method for
/// \arg \c program.
///
/// If a fallback method determines the function exists but the address is
/// not available PI_FUNCTION_ADDRESS_IS_NOT_AVAILABLE is returned. If the
/// address does not exist PI_INVALID_KERNEL_NAME is returned.
__SYCL_EXPORT pi_result piextGetDeviceFunctionPointer(
    pi_device device, pi_program program, const char *function_name,
    pi_uint64 *function_pointer_ret);

//
// Context
//
__SYCL_EXPORT pi_result piContextCreate(
    const pi_context_properties *properties, pi_uint32 num_devices,
    const pi_device *devices,
    void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb,
                       void *user_data),
    void *user_data, pi_context *ret_context);

__SYCL_EXPORT pi_result piContextGetInfo(pi_context context,
                                         pi_context_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piContextRetain(pi_context context);

__SYCL_EXPORT pi_result piContextRelease(pi_context context);

typedef void (*pi_context_extended_deleter)(void *user_data);

__SYCL_EXPORT pi_result piextContextSetExtendedDeleter(
    pi_context context, pi_context_extended_deleter func, void *user_data);

/// Gets the native handle of a PI context object.
///
/// \param context is the PI context to get the native handle of.
/// \param nativeHandle is the native handle of context.
__SYCL_EXPORT pi_result
piextContextGetNativeHandle(pi_context context, pi_native_handle *nativeHandle);

/// Creates PI context object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
/// NOTE: The number of devices and the list of devices is needed for Level Zero
/// backend because there is no possilibity to query this information from
/// context handle for Level Zero. If backend has API to query a list of devices
/// from the context native handle then these parameters are ignored.
///
/// \param nativeHandle is the native handle to create PI context from.
/// \param numDevices is the number of devices in the context. Parameter is
///        ignored if number of devices can be queried from the context native
///        handle for a backend.
/// \param devices is the list of devices in the context. Parameter is ignored
///        if devices can be queried from the context native handle for a
///        backend.
/// \param pluginOwnsNativeHandle Indicates whether the created PI object
///        should take ownership of the native handle.
/// \param context is the PI context created from the native handle.
/// \return PI_SUCCESS if successfully created pi_context from the handle.
///         PI_OUT_OF_HOST_MEMORY if can't allocate memory for the pi_context
///         object. PI_INVALID_VALUE if numDevices == 0 or devices is NULL but
///         backend doesn't have API to query a list of devices from the context
///         native handle. PI_UNKNOWN_ERROR in case of another error.
__SYCL_EXPORT pi_result piextContextCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_uint32 numDevices,
    const pi_device *devices, bool pluginOwnsNativeHandle, pi_context *context);

//
// Queue
//
__SYCL_EXPORT pi_result piQueueCreate(pi_context context, pi_device device,
                                      pi_queue_properties properties,
                                      pi_queue *queue);

__SYCL_EXPORT pi_result piQueueGetInfo(pi_queue command_queue,
                                       pi_queue_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piQueueRetain(pi_queue command_queue);

__SYCL_EXPORT pi_result piQueueRelease(pi_queue command_queue);

__SYCL_EXPORT pi_result piQueueFinish(pi_queue command_queue);

__SYCL_EXPORT pi_result piQueueFlush(pi_queue command_queue);

/// Gets the native handle of a PI queue object.
///
/// \param queue is the PI queue to get the native handle of.
/// \param nativeHandle is the native handle of queue.
__SYCL_EXPORT pi_result
piextQueueGetNativeHandle(pi_queue queue, pi_native_handle *nativeHandle);

/// Creates PI queue object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI queue from.
/// \param context is the PI context of the queue.
/// \param queue is the PI queue created from the native handle.
/// \param pluginOwnsNativeHandle Indicates whether the created PI object
///        should take ownership of the native handle.
__SYCL_EXPORT pi_result piextQueueCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, pi_queue *queue,
    bool pluginOwnsNativeHandle);

//
// Memory
//
__SYCL_EXPORT pi_result piMemBufferCreate(
    pi_context context, pi_mem_flags flags, size_t size, void *host_ptr,
    pi_mem *ret_mem, const pi_mem_properties *properties = nullptr);

__SYCL_EXPORT pi_result piMemImageCreate(pi_context context, pi_mem_flags flags,
                                         const pi_image_format *image_format,
                                         const pi_image_desc *image_desc,
                                         void *host_ptr, pi_mem *ret_mem);

__SYCL_EXPORT pi_result piMemGetInfo(
    pi_mem mem,
    cl_mem_info param_name, // TODO: untie from OpenCL
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piMemImageGetInfo(pi_mem image,
                                          pi_image_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piMemRetain(pi_mem mem);

__SYCL_EXPORT pi_result piMemRelease(pi_mem mem);

__SYCL_EXPORT pi_result piMemBufferPartition(
    pi_mem buffer, pi_mem_flags flags, pi_buffer_create_type buffer_create_type,
    void *buffer_create_info, pi_mem *ret_mem);

/// Gets the native handle of a PI mem object.
///
/// \param mem is the PI mem to get the native handle of.
/// \param nativeHandle is the native handle of mem.
__SYCL_EXPORT pi_result piextMemGetNativeHandle(pi_mem mem,
                                                pi_native_handle *nativeHandle);

/// Creates PI mem object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI mem from.
/// \param mem is the PI mem created from the native handle.
__SYCL_EXPORT pi_result
piextMemCreateWithNativeHandle(pi_native_handle nativeHandle, pi_mem *mem);

//
// Program
//

__SYCL_EXPORT pi_result piProgramCreate(pi_context context, const void *il,
                                        size_t length, pi_program *res_program);

__SYCL_EXPORT pi_result piclProgramCreateWithSource(pi_context context,
                                                    pi_uint32 count,
                                                    const char **strings,
                                                    const size_t *lengths,
                                                    pi_program *ret_program);

/// Creates a PI program for a context and loads the given binary into it.
///
/// \param context is the PI context to associate the program with.
/// \param num_devices is the number of devices in device_list.
/// \param device_list is a pointer to a list of devices. These devices must all
///                    be in context.
/// \param lengths is an array of sizes in bytes of the binary in binaries.
/// \param binaries is a pointer to a list of program binaries.
/// \param num_metadata_entries is the number of metadata entries in metadata.
/// \param metadata is a pointer to a list of program metadata entries. The
///                 use of metadata entries is backend-defined.
/// \param binary_status returns whether the program binary was loaded
///                      succesfully or not, for each device in device_list.
///                      binary_status is ignored if it is null and otherwise
///                      it must be an array of num_devices elements.
/// \param ret_program is the PI program created from the program binaries.
__SYCL_EXPORT pi_result piProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *ret_program);

__SYCL_EXPORT pi_result piProgramGetInfo(pi_program program,
                                         pi_program_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret);

__SYCL_EXPORT pi_result
piProgramLink(pi_context context, pi_uint32 num_devices,
              const pi_device *device_list, const char *options,
              pi_uint32 num_input_programs, const pi_program *input_programs,
              void (*pfn_notify)(pi_program program, void *user_data),
              void *user_data, pi_program *ret_program);

__SYCL_EXPORT pi_result piProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data);

__SYCL_EXPORT pi_result piProgramBuild(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data);

__SYCL_EXPORT pi_result piProgramGetBuildInfo(
    pi_program program, pi_device device,
    cl_program_build_info param_name, // TODO: untie from OpenCL
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piProgramRetain(pi_program program);

__SYCL_EXPORT pi_result piProgramRelease(pi_program program);

/// Sets a specialization constant to a specific value.
///
/// \param prog the program object which will use the value
/// \param spec_id integer ID of the constant
/// \param spec_size size of the value
/// \param spec_value bytes of the value
__SYCL_EXPORT pi_result
piextProgramSetSpecializationConstant(pi_program prog, pi_uint32 spec_id,
                                      size_t spec_size, const void *spec_value);

/// Gets the native handle of a PI program object.
///
/// \param program is the PI program to get the native handle of.
/// \param nativeHandle is the native handle of program.
__SYCL_EXPORT pi_result
piextProgramGetNativeHandle(pi_program program, pi_native_handle *nativeHandle);

/// Creates PI program object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI program from.
/// \param context is the PI context of the program.
/// \param pluginOwnsNativeHandle Indicates whether the created PI object
///        should take ownership of the native handle.
/// \param program is the PI program created from the native handle.
__SYCL_EXPORT pi_result piextProgramCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context,
    bool pluginOwnsNativeHandle, pi_program *program);

//
// Kernel
//

typedef enum {
  /// indicates that the kernel might access data through USM ptrs
  PI_USM_INDIRECT_ACCESS,
  /// provides an explicit list of pointers that the kernel will access
  PI_USM_PTRS = CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL
} _pi_kernel_exec_info;

using pi_kernel_exec_info = _pi_kernel_exec_info;

__SYCL_EXPORT pi_result piKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel);

__SYCL_EXPORT pi_result piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                       size_t arg_size, const void *arg_value);

__SYCL_EXPORT pi_result piKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                             pi_kernel_group_info param_name,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret);

/// API to query information from the sub-group from a kernel
///
/// \param kernel is the pi_kernel to query
/// \param device is the device the kernel is executed on
/// \param param_name is a pi_kernel_sub_group_info enum value that
///        specifies the informtation queried for.
/// \param input_value_size is the size of input value passed in
///        ptr input_value param
/// \param input_value is the ptr to the input value passed.
/// \param param_value_size is the size of the value in bytes.
/// \param param_value is a pointer to the value to set.
/// \param param_value_size_ret is a pointer to return the size of data in
///        param_value ptr.
///
/// All queries expect a return of 4 bytes in param_value_size,
/// param_value_size_ret, and a uint32_t value should to be written in
/// param_value ptr.
/// Note: This behaviour differs from OpenCL. OpenCL returns size_t.
__SYCL_EXPORT pi_result piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device, pi_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piKernelRetain(pi_kernel kernel);

__SYCL_EXPORT pi_result piKernelRelease(pi_kernel kernel);

/// Sets up pointer arguments for CL kernels. An extra indirection
/// is required due to CL argument conventions.
///
/// \param kernel is the kernel to be launched
/// \param arg_index is the index of the kernel argument
/// \param arg_size is the size in bytes of the argument (ignored in CL)
/// \param arg_value is the pointer argument
__SYCL_EXPORT pi_result piextKernelSetArgPointer(pi_kernel kernel,
                                                 pi_uint32 arg_index,
                                                 size_t arg_size,
                                                 const void *arg_value);

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
__SYCL_EXPORT pi_result piKernelSetExecInfo(pi_kernel kernel,
                                            pi_kernel_exec_info value_name,
                                            size_t param_value_size,
                                            const void *param_value);

/// Creates PI kernel object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI kernel from.
/// \param context is the PI context of the kernel.
/// \param program is the PI program of the kernel.
/// \param pluginOwnsNativeHandle Indicates whether the created PI object
///        should take ownership of the native handle.
/// \param kernel is the PI kernel created from the native handle.
__SYCL_EXPORT pi_result piextKernelCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, pi_program program,
    bool pluginOwnsNativeHandle, pi_kernel *kernel);

/// Gets the native handle of a PI kernel object.
///
/// \param kernel is the PI kernel to get the native handle of.
/// \param nativeHandle is the native handle of kernel.
__SYCL_EXPORT pi_result
piextKernelGetNativeHandle(pi_kernel kernel, pi_native_handle *nativeHandle);

//
// Events
//
__SYCL_EXPORT pi_result piEventCreate(pi_context context, pi_event *ret_event);

__SYCL_EXPORT pi_result piEventGetInfo(pi_event event, pi_event_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piEventGetProfilingInfo(pi_event event,
                                                pi_profiling_info param_name,
                                                size_t param_value_size,
                                                void *param_value,
                                                size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list);

__SYCL_EXPORT pi_result piEventSetCallback(
    pi_event event, pi_int32 command_exec_callback_type,
    void (*pfn_notify)(pi_event event, pi_int32 event_command_status,
                       void *user_data),
    void *user_data);

__SYCL_EXPORT pi_result piEventSetStatus(pi_event event,
                                         pi_int32 execution_status);

__SYCL_EXPORT pi_result piEventRetain(pi_event event);

__SYCL_EXPORT pi_result piEventRelease(pi_event event);

/// Gets the native handle of a PI event object.
///
/// \param event is the PI event to get the native handle of.
/// \param nativeHandle is the native handle of event.
__SYCL_EXPORT pi_result
piextEventGetNativeHandle(pi_event event, pi_native_handle *nativeHandle);

/// Creates PI event object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI event from.
/// \param context is the corresponding PI context
/// \param pluginOwnsNativeHandle Indicates whether the created PI object
///        should take ownership of the native handle.
/// \param event is the PI event created from the native handle.
__SYCL_EXPORT pi_result piextEventCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, bool ownNativeHandle,
    pi_event *event);

//
// Sampler
//
__SYCL_EXPORT pi_result piSamplerCreate(
    pi_context context, const pi_sampler_properties *sampler_properties,
    pi_sampler *result_sampler);

__SYCL_EXPORT pi_result piSamplerGetInfo(pi_sampler sampler,
                                         pi_sampler_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret);

__SYCL_EXPORT pi_result piSamplerRetain(pi_sampler sampler);

__SYCL_EXPORT pi_result piSamplerRelease(pi_sampler sampler);

//
// Queue Commands
//
__SYCL_EXPORT pi_result piEnqueueKernelLaunch(
    pi_queue queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueNativeKernel(
    pi_queue queue, void (*user_func)(void *), void *args, size_t cb_args,
    pi_uint32 num_mem_objects, const pi_mem *mem_list,
    const void **args_mem_loc, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueEventsWait(pi_queue command_queue,
                                            pi_uint32 num_events_in_wait_list,
                                            const pi_event *event_wait_list,
                                            pi_event *event);

__SYCL_EXPORT pi_result piEnqueueEventsWaitWithBarrier(
    pi_queue command_queue, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemBufferRead(
    pi_queue queue, pi_mem buffer, pi_bool blocking_read, size_t offset,
    size_t size, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result
piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                        pi_bool blocking_write, size_t offset, size_t size,
                        const void *ptr, pi_uint32 num_events_in_wait_list,
                        const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result
piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                       pi_mem dst_buffer, size_t src_offset, size_t dst_offset,
                       size_t size, pi_uint32 num_events_in_wait_list,
                       const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event);

__SYCL_EXPORT pi_result
piEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
                       const void *pattern, size_t pattern_size, size_t offset,
                       size_t size, pi_uint32 num_events_in_wait_list,
                       const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemImageRead(
    pi_queue command_queue, pi_mem image, pi_bool blocking_read,
    pi_image_offset origin, pi_image_region region, size_t row_pitch,
    size_t slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemImageWrite(
    pi_queue command_queue, pi_mem image, pi_bool blocking_write,
    pi_image_offset origin, pi_image_region region, size_t input_row_pitch,
    size_t input_slice_pitch, const void *ptr,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemImageCopy(
    pi_queue command_queue, pi_mem src_image, pi_mem dst_image,
    pi_image_offset src_origin, pi_image_offset dst_origin,
    pi_image_region region, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result
piEnqueueMemImageFill(pi_queue command_queue, pi_mem image,
                      const void *fill_color, const size_t *origin,
                      const size_t *region, pi_uint32 num_events_in_wait_list,
                      const pi_event *event_wait_list, pi_event *event);

__SYCL_EXPORT pi_result piEnqueueMemBufferMap(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    pi_map_flags map_flags, size_t offset, size_t size,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event, void **ret_map);

__SYCL_EXPORT pi_result piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                          void *mapped_ptr,
                                          pi_uint32 num_events_in_wait_list,
                                          const pi_event *event_wait_list,
                                          pi_event *event);

// Extension to allow backends to process a PI memory object before adding it
// as an argument for a kernel.
// Note: This is needed by the CUDA backend to extract the device pointer to
// the memory as the kernels uses it rather than the PI object itself.
__SYCL_EXPORT pi_result piextKernelSetArgMemObj(pi_kernel kernel,
                                                pi_uint32 arg_index,
                                                const pi_mem *arg_value);

// Extension to allow backends to process a PI sampler object before adding it
// as an argument for a kernel.
// Note: This is needed by the CUDA backend to extract the properties of the
// sampler as the kernels uses it rather than the PI object itself.
__SYCL_EXPORT pi_result piextKernelSetArgSampler(pi_kernel kernel,
                                                 pi_uint32 arg_index,
                                                 const pi_sampler *arg_value);

///
// USM
///
typedef enum {
  PI_USM_HOST_SUPPORT = CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
  PI_USM_DEVICE_SUPPORT = CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
  PI_USM_SINGLE_SHARED_SUPPORT =
      CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
  PI_USM_CROSS_SHARED_SUPPORT =
      CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
  PI_USM_SYSTEM_SHARED_SUPPORT = CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL
} _pi_usm_capability_query;

typedef enum : pi_bitfield {
  PI_USM_ACCESS = CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL,
  PI_USM_ATOMIC_ACCESS = CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL,
  PI_USM_CONCURRENT_ACCESS = CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL,
  PI_USM_CONCURRENT_ATOMIC_ACCESS =
      CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL
} _pi_usm_capabilities;

typedef enum {
  PI_MEM_ALLOC_TYPE = CL_MEM_ALLOC_TYPE_INTEL,
  PI_MEM_ALLOC_BASE_PTR = CL_MEM_ALLOC_BASE_PTR_INTEL,
  PI_MEM_ALLOC_SIZE = CL_MEM_ALLOC_SIZE_INTEL,
  PI_MEM_ALLOC_DEVICE = CL_MEM_ALLOC_DEVICE_INTEL,
} _pi_mem_info;

typedef enum {
  PI_MEM_TYPE_UNKNOWN = CL_MEM_TYPE_UNKNOWN_INTEL,
  PI_MEM_TYPE_HOST = CL_MEM_TYPE_HOST_INTEL,
  PI_MEM_TYPE_DEVICE = CL_MEM_TYPE_DEVICE_INTEL,
  PI_MEM_TYPE_SHARED = CL_MEM_TYPE_SHARED_INTEL
} _pi_usm_type;

typedef enum : pi_bitfield {
  PI_MEM_ALLOC_FLAGS = CL_MEM_ALLOC_FLAGS_INTEL
} _pi_usm_mem_properties;

// Flag is used for piProgramUSMEnqueuePrefetch. PI_USM_MIGRATION_TBD0 is a
// placeholder for future developments and should not change the behaviour of
// piProgramUSMEnqueuePrefetch
typedef enum : pi_bitfield {
  PI_USM_MIGRATION_TBD0 = (1 << 0)
} _pi_usm_migration_flags;

using pi_usm_capability_query = _pi_usm_capability_query;
using pi_usm_capabilities = _pi_usm_capabilities;
using pi_mem_info = _pi_mem_info;
using pi_usm_type = _pi_usm_type;
using pi_usm_mem_properties = _pi_usm_mem_properties;
using pi_usm_migration_flags = _pi_usm_migration_flags;

/// Allocates host memory accessible by the device.
///
/// \param result_ptr contains the allocated memory
/// \param context is the pi_context
/// \param properties are optional allocation properties
/// \param size is the size of the allocation
/// \param alignment is the desired alignment of the allocation
__SYCL_EXPORT pi_result piextUSMHostAlloc(void **result_ptr, pi_context context,
                                          pi_usm_mem_properties *properties,
                                          size_t size, pi_uint32 alignment);

/// Allocates device memory
///
/// \param result_ptr contains the allocated memory
/// \param context is the pi_context
/// \param device is the device the memory will be allocated on
/// \param properties are optional allocation properties
/// \param size is the size of the allocation
/// \param alignment is the desired alignment of the allocation
__SYCL_EXPORT pi_result piextUSMDeviceAlloc(void **result_ptr,
                                            pi_context context,
                                            pi_device device,
                                            pi_usm_mem_properties *properties,
                                            size_t size, pi_uint32 alignment);

/// Allocates memory accessible on both host and device
///
/// \param result_ptr contains the allocated memory
/// \param context is the pi_context
/// \param device is the device the memory will be allocated on
/// \param properties are optional allocation properties
/// \param size is the size of the allocation
/// \param alignment is the desired alignment of the allocation
__SYCL_EXPORT pi_result piextUSMSharedAlloc(void **result_ptr,
                                            pi_context context,
                                            pi_device device,
                                            pi_usm_mem_properties *properties,
                                            size_t size, pi_uint32 alignment);

/// Indicates that the allocated USM memory is no longer needed on the runtime
/// side. The actual freeing of the memory may be done in a blocking or deferred
/// manner, e.g. to avoid issues with indirect memory access from kernels.
///
/// \param context is the pi_context of the allocation
/// \param ptr is the memory to be freed
__SYCL_EXPORT pi_result piextUSMFree(pi_context context, void *ptr);

/// USM Memset API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to memset
/// \param value is value to set.  It is interpreted as an 8-bit value and the
/// upper
///        24 bits are ignored
/// \param count is the size in bytes to memset
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemset(pi_queue queue, void *ptr,
                                              pi_int32 value, size_t count,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event);

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
__SYCL_EXPORT pi_result piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                              void *dst_ptr,
                                              const void *src_ptr, size_t size,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event);

/// Hint to migrate memory to the device
///
/// \param queue is the queue to submit to
/// \param ptr points to the memory to migrate
/// \param size is the number of bytes to migrate
/// \param flags is a bitfield used to specify memory migration options
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueuePrefetch(
    pi_queue queue, const void *ptr, size_t size, pi_usm_migration_flags flags,
    pi_uint32 num_events_in_waitlist, const pi_event *events_waitlist,
    pi_event *event);

/// USM Memadvise API
///
/// \param queue is the queue to submit to
/// \param ptr is the data to be advised
/// \param length is the size in bytes of the memory to advise
/// \param advice is device specific advice
/// \param event is the event that represents this operation
// USM memadvise API to govern behavior of automatic migration mechanisms
__SYCL_EXPORT pi_result piextUSMEnqueueMemAdvise(pi_queue queue,
                                                 const void *ptr, size_t length,
                                                 pi_mem_advice advice,
                                                 pi_event *event);

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
/// \param param_value_size_ret is how many bytes were written
__SYCL_EXPORT pi_result piextUSMGetMemAllocInfo(
    pi_context context, const void *ptr, pi_mem_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

/// API to get Plugin internal data, opaque to SYCL RT. Some devices whose
/// device code is compiled by the host compiler (e.g. CPU emulators) may use it
/// to access some device code functionality implemented in/behind the plugin.
/// \param opaque_data_param - unspecified argument, interpretation is specific
/// to a plugin \param opaque_data_return - placeholder for the returned opaque
/// data.
__SYCL_EXPORT pi_result piextPluginGetOpaqueData(void *opaque_data_param,
                                                 void **opaque_data_return);

/// API to notify that the plugin should clean up its resources.
/// No PI calls should be made until the next piPluginInit call.
/// \param PluginParameter placeholder for future use, currenly not used.
__SYCL_EXPORT pi_result piTearDown(void *PluginParameter);

struct _pi_plugin {
  // PI version supported by host passed to the plugin. The Plugin
  // checks and writes the appropriate Function Pointers in
  // PiFunctionTable.
  // TODO: Work on version fields and their handshaking mechanism.
  // Some choices are:
  // - Use of integers to keep major and minor version.
  // - Keeping char* Versions.
  char PiVersion[4];
  // Plugin edits this.
  char PluginVersion[4];
  char *Targets;
  struct FunctionPointers {
#define _PI_API(api) decltype(::api) *api;
#include <CL/sycl/detail/pi.def>
  } PiFunctionTable;
};

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // _PI_H_

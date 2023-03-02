//===--------- ur_cuda.cpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <cassert>
#include <sstream>

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <sycl/detail/defines.hpp>

#include "ur_cuda.hpp"
#include <ur/ur.hpp>
#include <ur_api.h>

/// ------ Error handling, matching OpenCL plugin semantics.
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
namespace ur {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message) {
  std::cerr << "pi_die: " << Message << std::endl;
  std::terminate();
}

// Reports error messages
void cuPrint(const char *Message) {
  std::cerr << "pi_print: " << Message << std::endl;
}

void assertion(bool Condition, const char *Message = nullptr) {
  if (!Condition)
    die(Message);
}

} // namespace pi
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

std::string getCudaVersionString() {
  int driver_version = 0;
  cuDriverGetVersion(&driver_version);
  // The version is returned as (1000 major + 10 minor).
  std::stringstream stream;
  stream << "CUDA " << driver_version / 1000 << "."
         << driver_version % 1000 / 10;
  return stream.str();
}


ur_result_t map_error(CUresult result) {
  switch (result) {
  case CUDA_SUCCESS:
    return UR_RESULT_SUCCESS;
  case CUDA_ERROR_NOT_PERMITTED:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  case CUDA_ERROR_INVALID_CONTEXT:
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  case CUDA_ERROR_INVALID_DEVICE:
    return UR_RESULT_ERROR_INVALID_DEVICE;
  case CUDA_ERROR_INVALID_VALUE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case CUDA_ERROR_OUT_OF_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

/// Converts CUDA error into PI error codes, and outputs error information
/// to stderr.
/// If PI_CUDA_ABORT env variable is defined, it aborts directly instead of
/// throwing the error. This is intended for debugging purposes.
/// \return PI_SUCCESS if \param result was CUDA_SUCCESS.
/// \throw pi_error exception (integer) if input was not success.
///
ur_result_t check_error(CUresult result, const char *function, int line,
                      const char *file) {
  if (result == CUDA_SUCCESS || result == CUDA_ERROR_DEINITIALIZED) {
    return UR_RESULT_SUCCESS;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    const char *errorString = nullptr;
    const char *errorName = nullptr;
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorString);
    std::stringstream ss;
    ss << "\nPI CUDA ERROR:"
       << "\n\tValue:           " << result
       << "\n\tName:            " << errorName
       << "\n\tDescription:     " << errorString
       << "\n\tFunction:        " << function << "\n\tSource Location: " << file
       << ":" << line << "\n"
       << std::endl;
    std::cerr << ss.str();
  }

  if (std::getenv("PI_CUDA_ABORT") != nullptr) {
    std::abort();
  }

  throw map_error(result);
}

/// \cond NODOXY
#define PI_CHECK_ERROR(result) check_error(result, __func__, __LINE__, __FILE__)

int getAttribute(ur_device_handle_t device, CUdevice_attribute attribute) {
  int value;
  sycl::detail::ur::assertion(
      cuDeviceGetAttribute(&value, attribute, device->get()) == CUDA_SUCCESS);
  return value;
}


class ScopedContext {
public:
  ScopedContext(CUcontext ctxt) { set_context(ctxt); }

  ~ScopedContext() {}

private:
  void set_context(CUcontext desired) {
    CUcontext original = nullptr;

    PI_CHECK_ERROR(cuCtxGetCurrent(&original));

    // Make sure the desired context is active on the current thread, setting
    // it if necessary
    if (original != desired) {
      PI_CHECK_ERROR(cuCtxSetCurrent(desired));
    }
  }
};

// ---------------------------------------------------------------------------

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t device,
                                                    ur_device_info_t infoType,
                                                    size_t propSize,
                                                    void *pDeviceInfo,
                                                    size_t *pPropSizeRet) {
  PI_ASSERT(device, UR_RESULT_ERROR_INVALID_DEVICE);
  UrReturnHelper ReturnValue(propSize, pDeviceInfo, pPropSizeRet);

  static constexpr uint32_t max_work_item_dimensions = 3u;

  assert(device != nullptr);

  ScopedContext active(device->get_context());

  switch ((uint32_t) infoType) {
  case UR_DEVICE_INFO_TYPE: {
    return ReturnValue(UR_DEVICE_TYPE_GPU);
  }
  case UR_DEVICE_INFO_VENDOR_ID: {
    return ReturnValue(4318u);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    int compute_units = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&compute_units,
                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(compute_units >= 0);
    return ReturnValue(static_cast<uint32_t>(compute_units));
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return ReturnValue(max_work_item_dimensions);
  }
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t sizes[max_work_item_dimensions];
    } return_sizes;

    int max_x = 0, max_y = 0, max_z = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(max_x >= 0);

    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(max_y >= 0);

    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(max_z >= 0);

    return_sizes.sizes[0] = size_t(max_x);
    return_sizes.sizes[1] = size_t(max_y);
    return_sizes.sizes[2] = size_t(max_z);
    return ReturnValue(return_sizes);
  }

  case UR_EXT_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t sizes[max_work_item_dimensions];
    } return_sizes;
    int max_x = 0, max_y = 0, max_z = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(max_x >= 0);

    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(max_y >= 0);

    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(max_z >= 0);

    return_sizes.sizes[0] = size_t(max_x);
    return_sizes.sizes[1] = size_t(max_y);
    return_sizes.sizes[2] = size_t(max_z);
    return ReturnValue(return_sizes);
  }

  case UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    int max_work_group_size = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_work_group_size,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             device->get()) == CUDA_SUCCESS);

    sycl::detail::ur::assertion(max_work_group_size >= 0);

    return ReturnValue(size_t(max_work_group_size));
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE: {
    return ReturnValue(1u);
  }
  case UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int max_threads = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_threads,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             device->get()) == CUDA_SUCCESS);
    int warpSize = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                             device->get()) == CUDA_SUCCESS);
    int maxWarps = (max_threads + warpSize - 1) / warpSize;
    return ReturnValue(maxWarps);
  }
  case UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // Volta provides independent thread scheduling
    // TODO: Revisit for previous generation GPUs
    int major = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             device->get()) == CUDA_SUCCESS);
    bool ifp = (major >= 7);
    return ReturnValue(ifp);
  }

  case UR_DEVICE_INFO_ATOMIC_64: {
    int major = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             device->get()) == CUDA_SUCCESS);

    bool atomic64 = (major >= 6) ? true : false;
    return ReturnValue(uint32_t{atomic64});
  }
  // TODO(UR): implement the two queries below when the UR commit is updated
  // to the newest version
  case UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {
    uint64_t capabilities =
        PI_MEMORY_ORDER_RELAXED | PI_MEMORY_ORDER_ACQUIRE |
        PI_MEMORY_ORDER_RELEASE | PI_MEMORY_ORDER_ACQ_REL;
    return ReturnValue(capabilities);
  }
  case UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES: {
    int major = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             device->get()) == CUDA_SUCCESS);
    uint64_t capabilities =
        (major >= 7) ? PI_MEMORY_SCOPE_WORK_ITEM | PI_MEMORY_SCOPE_SUB_GROUP
        |
                           PI_MEMORY_SCOPE_WORK_GROUP |
                           PI_MEMORY_SCOPE_DEVICE | PI_MEMORY_SCOPE_SYSTEM
                     : PI_MEMORY_SCOPE_WORK_ITEM | PI_MEMORY_SCOPE_SUB_GROUP
                     |
                           PI_MEMORY_SCOPE_WORK_GROUP |
                           PI_MEMORY_SCOPE_DEVICE;
    return ReturnValue(capabilities);
  }
  case UR_DEVICE_INFO_BFLOAT16: {
    int major = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             device->get()) == CUDA_SUCCESS);

    bool bfloat16 = (major >= 8) ? true : false;
    return ReturnValue(bfloat16);
  }
  case UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    // NVIDIA devices only support one sub-group size (the warp size)
    int warpSize = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                             device->get()) == CUDA_SUCCESS);
    size_t sizes[1] = {static_cast<size_t>(warpSize)};
    return ReturnValue(sizes, 1);
  }
  case UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    int clock_freq = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&clock_freq, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(clock_freq >= 0);
    return ReturnValue(static_cast<uint32_t>(clock_freq) / 1000u);
  }
  case UR_DEVICE_INFO_ADDRESS_BITS: {
    auto bits = uint32_t{std::numeric_limits<uintptr_t>::digits};
    return ReturnValue(bits);
  }
  case UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    // Max size of memory object allocation in bytes.
    // The minimum value is max(min(1024 × 1024 ×
    // 1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE),
    // 32 × 1024 × 1024) for devices that are not of type
    // CL_DEVICE_TYPE_CUSTOM.

    size_t global = 0;
    sycl::detail::ur::assertion(cuDeviceTotalMem(&global, device->get()) ==
                                CUDA_SUCCESS);

    auto quarter_global = static_cast<uint32_t>(global / 4u);

    auto max_alloc = std::max(std::min(1024u * 1024u * 1024u, quarter_global),
                              32u * 1024u * 1024u);

    return ReturnValue(uint64_t{max_alloc});
  }
  case UR_DEVICE_INFO_IMAGE_SUPPORTED: {
    bool enabled = false;

    if (std::getenv("SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT") != nullptr) {
      enabled = true;
    } else {
      sycl::detail::ur::cuPrint(
          "Images are not fully supported by the CUDA BE, their support is "
          "disabled by default. Their partial support can be activated by "
          "setting SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT environment variable at "
          "runtime.");
    }

    return ReturnValue(uint32_t{enabled});
  }
  case UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    // This call doesn't match to CUDA as it doesn't have images, but instead
    // surfaces and textures. No clear call in the CUDA API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    // This call doesn't match to CUDA as it doesn't have images, but instead
    // surfaces and textures. No clear call in the CUDA API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int tex_height = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&tex_height,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(tex_height >= 0);
    int surf_height = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&surf_height,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(surf_height >= 0);

    int min = std::min(tex_height, surf_height);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&tex_width,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(tex_width >= 0);
    int surf_width = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&surf_width,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int tex_height = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&tex_height,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(tex_height >= 0);
    int surf_height = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&surf_height,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(surf_height >= 0);

    int min = std::min(tex_height, surf_height);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&tex_width,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(tex_width >= 0);
    int surf_width = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&surf_width,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    // Take the smaller of maximum surface and maximum texture depth.
    int tex_depth = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&tex_depth,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(tex_depth >= 0);
    int surf_depth = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&surf_depth,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(surf_depth >= 0);

    int min = std::min(tex_depth, surf_depth);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&tex_width,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(tex_width >= 0);
    int surf_width = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&surf_width,
                             CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return ReturnValue(static_cast<size_t>(min));
  }
  case UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    return ReturnValue(0lu);
  }
  case UR_DEVICE_INFO_MAX_SAMPLERS: {
    // This call is kind of meaningless for cuda, as samplers don't exist.
    // Closest thing is textures, which is 128.
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
    // __global__ function parameters are passed to the device via constant
    // memory and are limited to 4 KB.
    return ReturnValue(4000lu);
  }
  case UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    int mem_base_addr_align = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&mem_base_addr_align,
                             CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                             device->get()) == CUDA_SUCCESS);
    // Multiply by 8 as clGetDeviceInfo returns this value in bits
    mem_base_addr_align *= 8;
    return ReturnValue(mem_base_addr_align);
  }
  case UR_DEVICE_INFO_HALF_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_SINGLE_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    uint64_t config =
        UR_FP_CAPABILITY_FLAG_DENORM | UR_FP_CAPABILITY_FLAG_INF_NAN |
        UR_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        UR_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        UR_FP_CAPABILITY_FLAG_ROUND_TO_INF | UR_FP_CAPABILITY_FLAG_FMA |
        UR_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    return ReturnValue(config);
  }
  case UR_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    uint64_t config =
        UR_FP_CAPABILITY_FLAG_DENORM | UR_FP_CAPABILITY_FLAG_INF_NAN |
        UR_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        UR_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        UR_FP_CAPABILITY_FLAG_ROUND_TO_INF | UR_FP_CAPABILITY_FLAG_FMA;
    return ReturnValue(config);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    return ReturnValue(UR_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // The value is documented for all existing GPUs in the CUDA programming
    // guidelines, section "H.3.2. Global Memory".
    return ReturnValue(128u);
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    int cache_size = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(cache_size >= 0);
    // The L2 cache is global to the GPU.
    return ReturnValue(static_cast<uint64_t>(cache_size));
  }
  case UR_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    size_t bytes = 0;
    // Runtime API has easy access to this value, driver API info is scarse.
    sycl::detail::ur::assertion(cuDeviceTotalMem(&bytes, device->get()) ==
                                CUDA_SUCCESS);
    return ReturnValue(uint64_t{bytes});
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    int constant_memory = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&constant_memory,
                             CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(constant_memory >= 0);

    return ReturnValue(static_cast<uint64_t>(constant_memory));
  }
  case UR_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: is there a way to retrieve this from CUDA driver API?
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
    // 1060 3GB
    return ReturnValue(9u);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return ReturnValue(UR_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  }
  case UR_DEVICE_INFO_LOCAL_MEM_SIZE: {
    // OpenCL's "local memory" maps most closely to CUDA's "shared memory".
    // CUDA has its own definition of "local memory", which maps to OpenCL's
    // "private memory".
    int local_mem_size = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&local_mem_size,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(local_mem_size >= 0);
    return ReturnValue(static_cast<uint64_t>(local_mem_size));
  }
  case UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    int ecc_enabled = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&ecc_enabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
                             device->get()) == CUDA_SUCCESS);

    sycl::detail::ur::assertion((ecc_enabled == 0) | (ecc_enabled == 1));
    auto result = static_cast<bool>(ecc_enabled);
    return ReturnValue(uint32_t{result});
  }
  case UR_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    int is_integrated = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&is_integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED,
                             device->get()) == CUDA_SUCCESS);

    sycl::detail::ur::assertion((is_integrated == 0) | (is_integrated == 1));
    auto result = static_cast<bool>(is_integrated);
    return ReturnValue(uint32_t{result});
  }
  case UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
    // 1060 3GB
    return ReturnValue(1000lu);
  }
  case UR_DEVICE_INFO_ENDIAN_LITTLE: {
    return ReturnValue(uint32_t{true});
  }
  case UR_DEVICE_INFO_AVAILABLE: {
    return ReturnValue(uint32_t{true});

  }
  case UR_EXT_DEVICE_INFO_BUILD_ON_SUBDEVICE: {
    return ReturnValue(uint32_t{true});

  }
  case UR_DEVICE_INFO_COMPILER_AVAILABLE: {
    return ReturnValue(uint32_t{true});

  }
  case UR_DEVICE_INFO_LINKER_AVAILABLE: {
    return ReturnValue(uint32_t{true});

  }
  case UR_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    auto capability = ur_device_exec_capability_flags_t{
        UR_DEVICE_EXEC_CAPABILITY_FLAG_KERNEL};
    return ReturnValue(capability);
  }
  case UR_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(
        ur_queue_flag_t(UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                        UR_QUEUE_FLAG_PROFILING_ENABLE));
  case UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    uint64_t capability = UR_QUEUE_FLAG_PROFILING_ENABLE |
                          UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return ReturnValue(capability);
  }
  case UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
    // The mandated minimum capability:
    uint64_t capability = UR_QUEUE_FLAG_PROFILING_ENABLE;
    return ReturnValue(capability);
  }
  case UR_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_PLATFORM: {
    return ReturnValue(device->get_platform());
  }
  case UR_DEVICE_INFO_NAME: {
    static constexpr size_t MAX_DEVICE_NAME_LENGTH = 256u;
    char name[MAX_DEVICE_NAME_LENGTH];
    sycl::detail::ur::assertion(cuDeviceGetName(name, MAX_DEVICE_NAME_LENGTH,
                                                device->get()) == CUDA_SUCCESS);
    return ReturnValue(name, strlen(name) + 1);
  }
  case UR_DEVICE_INFO_VENDOR: {
    return ReturnValue("NVIDIA Corporation");
  }
  case UR_DEVICE_INFO_DRIVER_VERSION: {
    auto version = getCudaVersionString();
    return ReturnValue(version.c_str());
  }
  case UR_DEVICE_INFO_PROFILE: {
    return ReturnValue("CUDA");
  }
  case UR_DEVICE_INFO_REFERENCE_COUNT: {
    return ReturnValue(device->get_reference_count());
  }
  case UR_DEVICE_INFO_VERSION: {
    return ReturnValue("PI 0.0");
  }
  case UR_EXT_DEVICE_INFO_OPENCL_C_VERSION: {
    return ReturnValue("");
  }
  case UR_DEVICE_INFO_EXTENSIONS: {

    std::string SupportedExtensions = "cl_khr_fp64 ";
    SupportedExtensions += PI_DEVICE_INFO_EXTENSION_DEVICELIB_ASSERT;
    SupportedExtensions += " ";

    int major = 0;
    int minor = 0;

    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&minor,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                             device->get()) == CUDA_SUCCESS);

    if ((major >= 6) || ((major == 5) && (minor >= 3))) {
      SupportedExtensions += "cl_khr_fp16 ";
    }

    return ReturnValue(SupportedExtensions.c_str());
  }
  case UR_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    // The minimum value for the FULL profile is 1 MB.
    return ReturnValue(1024lu);
  }
  case UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    return ReturnValue(uint32_t{true});
  }
  case UR_DEVICE_INFO_PARENT_DEVICE: {
    return ReturnValue(nullptr);
  }
  case UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_PROPERTIES: {
    return ReturnValue(static_cast<ur_device_partition_t>(0u));
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return ReturnValue(0u);
  }
  case UR_DEVICE_INFO_PARTITION_TYPE: {
    return ReturnValue(static_cast<ur_device_partition_t>(0u));
  }

    // Intel USM extensions

  case UR_DEVICE_INFO_USM_HOST_SUPPORT: {
    // from cl_intel_unified_shared_memory: "The host memory access capabilities
    // apply to any host allocation."
    //
    // query if/how the device can access page-locked host memory, possibly
    // through PCIe, using the same pointer as the host
    uint64_t value = {};
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)) {
      // the device shares a unified address space with the host
      if (getAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >=
          6) {
        // compute capability 6.x introduces operations that are atomic with
        // respect to other CPUs and GPUs in the system
        value = UR_EXT_USM_CAPS_ACCESS | UR_EXT_USM_CAPS_ATOMIC_ACCESS |
                UR_EXT_USM_CAPS_CONCURRENT_ACCESS |
                UR_EXT_USM_CAPS_CONCURRENT_ATOMIC_ACCESS;
      } else {
        // on GPU architectures with compute capability lower than 6.x, atomic
        // operations from the GPU to CPU memory will not be atomic with respect
        // to CPU initiated atomic operations
        value = UR_EXT_USM_CAPS_ACCESS | UR_EXT_USM_CAPS_CONCURRENT_ACCESS;
      }
    }
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    uint64_t value = UR_EXT_USM_CAPS_ACCESS | UR_EXT_USM_CAPS_ATOMIC_ACCESS |
                     UR_EXT_USM_CAPS_CONCURRENT_ACCESS |
                     UR_EXT_USM_CAPS_CONCURRENT_ATOMIC_ACCESS;
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The single device shared memory access capabilities apply to any shared
    // allocation associated with this device."
    //
    // query if/how the device can access managed memory associated to it
    uint64_t value = {};
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)) {
      // the device can allocate managed memory on this system
      value = UR_EXT_USM_CAPS_ACCESS | UR_EXT_USM_CAPS_ATOMIC_ACCESS;
    }
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
      // the device can coherently access managed memory concurrently with the
      // CPU
      value |= UR_EXT_USM_CAPS_CONCURRENT_ACCESS;
      if (getAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >=
          6) {
        // compute capability 6.x introduces operations that are atomic with
        // respect to other CPUs and GPUs in the system
        value |= UR_EXT_USM_CAPS_CONCURRENT_ATOMIC_ACCESS;
      }
    }
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The cross-device shared memory access capabilities apply to any shared
    // allocation associated with this device, or to any shared memory
    // allocation on another device that also supports the same cross-device
    // shared memory access capability."
    //
    // query if/how the device can access managed memory associated to other
    // devices
    uint64_t value = {};
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)) {
      // the device can allocate managed memory on this system
      value |= UR_EXT_USM_CAPS_ACCESS;
    }
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
      // all devices with the CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
      // attribute can coherently access managed memory concurrently with the
      // CPU
      value |= UR_EXT_USM_CAPS_CONCURRENT_ACCESS;
    }
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >=
        6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      if (value & UR_EXT_USM_CAPS_ACCESS)
        value |= UR_EXT_USM_CAPS_ATOMIC_ACCESS;
      if (value & UR_EXT_USM_CAPS_CONCURRENT_ACCESS)
        value |= UR_EXT_USM_CAPS_CONCURRENT_ATOMIC_ACCESS;
    }
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The shared system memory access capabilities apply to any allocations
    // made by a system allocator, such as malloc or new."
    //
    // query if/how the device can access pageable host memory allocated by the
    // system allocator
    uint64_t value = {};
    if (getAttribute(device, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)) {
      // the device suppports coherently accessing pageable memory without
      // calling cuMemHostRegister/cudaHostRegister on it
      if (getAttribute(device,
                       CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED)) {
        // the link between the device and the host supports native atomic
        // operations
        value = UR_EXT_USM_CAPS_ACCESS | UR_EXT_USM_CAPS_ATOMIC_ACCESS |
                UR_EXT_USM_CAPS_CONCURRENT_ACCESS |
                UR_EXT_USM_CAPS_CONCURRENT_ATOMIC_ACCESS;
      } else {
        // the link between the device and the host does not support native
        // atomic operations
        value = UR_EXT_USM_CAPS_ACCESS | UR_EXT_USM_CAPS_CONCURRENT_ACCESS;
      }
    }
    return ReturnValue(value);
  }
    // TODO(UR): Implement the below queries once the latest version of UR is
    // used 
    case UR_EXT_DEVICE_INFO_CUDA_ASYNC_BARRIER: {
      int value =
          getAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
          >= 8;
      return ReturnValue(value);
    }
    case UR_EXT_DEVICE_INFO_BACKEND_VERSION: {
      int major =
          getAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
      int minor =
          getAttribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
      std::string result = std::to_string(major) + "." +
      std::to_string(minor);
      return ReturnValue(result.c_str());
    }

  case UR_EXT_DEVICE_INFO_FREE_MEMORY: {
    size_t FreeMemory = 0;
    size_t TotalMemory = 0;
    sycl::detail::ur::assertion(cuMemGetInfo(&FreeMemory, &TotalMemory) ==
                                    CUDA_SUCCESS,
                                "failed cuMemGetInfo() API.");
    return ReturnValue(FreeMemory);
  }
  case UR_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    int value = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(value >= 0);
    // Convert kilohertz to megahertz when returning.
    return ReturnValue(value / 1000);
  }
  case UR_EXT_DEVICE_INFO_MEMORY_BUS_WIDTH: {
    int value = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&value,
                             CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(value >= 0);
    return ReturnValue(value);
  }
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    return ReturnValue(int32_t{1});
  }
  case UR_DEVICE_INFO_DEVICE_ID: {
    int value = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                             device->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(value >= 0);
    return ReturnValue(value);
  }
  case PI_DEVICE_INFO_UUID: {
    int driver_version = 0;
    cuDriverGetVersion(&driver_version);
    int major = driver_version / 1000;
    int minor = driver_version % 1000 / 10;
    CUuuid uuid;
    if ((major > 11) || (major == 11 && minor >= 4)) {
      sycl::detail::ur::assertion(cuDeviceGetUuid_v2(&uuid, device->get()) ==
                                  CUDA_SUCCESS);
    } else {
      sycl::detail::ur::assertion(cuDeviceGetUuid(&uuid, device->get()) ==
                                  CUDA_SUCCESS);
    }
    std::array<unsigned char, 16> name;
    std::copy(uuid.bytes, uuid.bytes + 16, name.begin());
    return ReturnValue(name.data(), 16);
  }
    // TODO: Investigate if this information is available on CUDA.
  case UR_DEVICE_INFO_PCI_ADDRESS:
  case UR_DEVICE_INFO_GPU_EU_COUNT:
  case UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case UR_EXT_DEVICE_INFO_GPU_SLICES:
  case UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case UR_EXT_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case UR_EXT_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
  case UR_EXT_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    return UR_RESULT_ERROR_INVALID_VALUE;

  default:
    break;
  }
  sycl::detail::ur::die("Device info request not implemented");
  return {};


  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPlatformGetInfo(
    ur_platform_handle_t hPlatform, ur_platform_info_t PlatformInfoType,
    size_t Size, void *pPlatformInfo, size_t *pSizeRet) {

  PI_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_PLATFORM);
  UrReturnHelper ReturnValue(Size, pPlatformInfo, pSizeRet);

  switch (PlatformInfoType) {
  case UR_PLATFORM_INFO_NAME:
    return ReturnValue("NVIDIA CUDA BACKEND");
  case UR_PLATFORM_INFO_VENDOR_NAME:
    return ReturnValue("NVIDIA Corporation");
  case UR_PLATFORM_INFO_PROFILE:
    return ReturnValue("FULL PROFILE");
  case UR_PLATFORM_INFO_VERSION: {
    auto version = getCudaVersionString();
    return ReturnValue(version.c_str()); // TODO(ur): Check this
  }
  case UR_PLATFORM_INFO_EXTENSIONS: {
    return ReturnValue("");
  }
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPlatformGet(uint32_t NumEntries, ur_platform_handle_t *phPlatforms,
              uint32_t *pNumPlatforms) {

  try {
    static std::once_flag initFlag;
    static uint32_t numPlatforms = 1;
    static std::vector<ur_platform_handle_t_> platformIds;

    if (phPlatforms == nullptr && pNumPlatforms == nullptr) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    ur_result_t err = UR_RESULT_SUCCESS;

    std::call_once(
        initFlag,
        [](ur_result_t &err) {
          if (cuInit(0) != CUDA_SUCCESS) {
            numPlatforms = 0;
            return;
          }
          int numDevices = 0;
          err = PI_CHECK_ERROR(cuDeviceGetCount(&numDevices));
          if (numDevices == 0) {
            numPlatforms = 0;
            return;
          }
          try {
            // make one platform per device
            numPlatforms = numDevices;
            platformIds.resize(numDevices);

            for (int i = 0; i < numDevices; ++i) {
              CUdevice device;
              err = PI_CHECK_ERROR(cuDeviceGet(&device, i));
              CUcontext context;
              err = PI_CHECK_ERROR(cuDevicePrimaryCtxRetain(&context, device));

              ScopedContext active(context);
              CUevent evBase;
              err = PI_CHECK_ERROR(cuEventCreate(&evBase, CU_EVENT_DEFAULT));

              // Use default stream to record base event counter
              err = PI_CHECK_ERROR(cuEventRecord(evBase, 0));

              platformIds[i].devices_.emplace_back(new ur_device_handle_t_{
                  device, context, evBase, &platformIds[i]});
              {
                const auto &dev = platformIds[i].devices_.back().get();
                size_t maxWorkGroupSize = 0u;
                size_t maxThreadsPerBlock[3] = {};
                ur_result_t retError = urDeviceGetInfo(
                    dev, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                    sizeof(maxThreadsPerBlock), maxThreadsPerBlock, nullptr);
                assert(retError == UR_RESULT_SUCCESS);
                (void)retError;

                retError = urDeviceGetInfo(
                    dev, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
                assert(retError == UR_RESULT_SUCCESS);

                dev->save_max_work_item_sizes(sizeof(maxThreadsPerBlock),
                                              maxThreadsPerBlock);
                dev->save_max_work_group_size(maxWorkGroupSize);
              }
            }
          } catch (const std::bad_alloc &) {
            // Signal out-of-memory situation
            for (int i = 0; i < numDevices; ++i) {
              platformIds[i].devices_.clear();
            }
            platformIds.clear();
            err = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
          } catch (...) {
            // Clear and rethrow to allow retry
            for (int i = 0; i < numDevices; ++i) {
              platformIds[i].devices_.clear();
            }
            platformIds.clear();
            throw;
          }
        },
        err);

    if (pNumPlatforms != nullptr) {
      *pNumPlatforms = numPlatforms;
    }

    if (phPlatforms != nullptr) {
      for (unsigned i = 0; i < std::min(NumEntries, numPlatforms); ++i) {
        phPlatforms[i] = &platformIds[i];
      }
    }

    return err;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

/// \return PI_SUCCESS if the function is executed successfully
/// CUDA devices are always root devices so retain always returns success.
UR_APIEXPORT ur_result_t UR_APICALL
urDeviceRetain(ur_device_handle_t device) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urDevicePartition(ur_device_handle_t, const ur_device_partition_property_t *,
                  uint32_t, ur_device_handle_t *, uint32_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// \return UR_RESULT_SUCCESS always since CUDA devices are always root
/// devices.
ur_result_t urDeviceRelease(ur_device_handle_t device) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urDeviceGet(ur_platform_handle_t hPlatform,
                                                ur_device_type_t DeviceType,
                                                uint32_t NumEntries,
                                                ur_device_handle_t *phDevices,
                                                uint32_t *pNumDevices) {
  ur_result_t err = UR_RESULT_SUCCESS;
  const bool askingForDefault = DeviceType == UR_DEVICE_TYPE_DEFAULT;
  const bool askingForGPU = DeviceType & UR_DEVICE_TYPE_GPU;
  const bool returnDevices = askingForDefault || askingForGPU;

  PI_ASSERT(hPlatform, UR_RESULT_ERROR_INVALID_PLATFORM);

  size_t numDevices = returnDevices ? hPlatform->devices_.size() : 0;

  try {
    if (!pNumDevices && !phDevices) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    if (pNumDevices) {
      *pNumDevices = numDevices;
    }

    if (returnDevices && phDevices) {
      for (size_t i = 0; i < std::min(size_t(NumEntries), numDevices); ++i) {
        phDevices[i] = hPlatform->devices_[i].get();
      }
    }

    return err;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

uint64_t ur_device_handle_t_::get_elapsed_time(CUevent ev) const {
  float miliSeconds = 0.0f;

  PI_CHECK_ERROR(cuEventElapsedTime(&miliSeconds, evBase_, ev));

  return static_cast<uint64_t>(miliSeconds * 1.0e6);
}

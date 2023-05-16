//==---------- pi_hip.cpp - HIP Plugin ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_hip.cpp
/// Implementation of HIP Plugin.
///
/// \ingroup sycl_pi_hip

#include <pi_hip.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/hip_definitions.hpp>
#include <sycl/detail/pi.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <string.h>
#include <string_view>

namespace {
pi_result map_error(hipError_t result) {
  switch (result) {
  case hipSuccess:
    return PI_SUCCESS;
  case hipErrorInvalidContext:
    return PI_ERROR_INVALID_CONTEXT;
  case hipErrorInvalidDevice:
    return PI_ERROR_INVALID_DEVICE;
  case hipErrorInvalidValue:
    return PI_ERROR_INVALID_VALUE;
  case hipErrorOutOfMemory:
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  case hipErrorLaunchOutOfResources:
    return PI_ERROR_OUT_OF_RESOURCES;
  default:
    return PI_ERROR_UNKNOWN;
  }
}

// TODO(ur) - this can be removed once more of pi entry points are ported to UR.
pi_result map_ur_error(ur_result_t result) {

  switch (result) {
#define CASE(UR_ERR, PI_ERR)                                                   \
  case UR_ERR:                                                                 \
    return PI_ERR;

    CASE(UR_RESULT_SUCCESS, PI_SUCCESS)
    CASE(UR_RESULT_ERROR_INVALID_OPERATION, PI_ERROR_INVALID_OPERATION)
    CASE(UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES,
         PI_ERROR_INVALID_QUEUE_PROPERTIES)
    CASE(UR_RESULT_ERROR_INVALID_QUEUE, PI_ERROR_INVALID_QUEUE)
    CASE(UR_RESULT_ERROR_INVALID_VALUE, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_CONTEXT, PI_ERROR_INVALID_CONTEXT)
    CASE(UR_RESULT_ERROR_INVALID_PLATFORM, PI_ERROR_INVALID_PLATFORM)
    CASE(UR_RESULT_ERROR_INVALID_BINARY, PI_ERROR_INVALID_BINARY)
    CASE(UR_RESULT_ERROR_INVALID_PROGRAM, PI_ERROR_INVALID_BINARY)
    CASE(UR_RESULT_ERROR_INVALID_SAMPLER, PI_ERROR_INVALID_SAMPLER)
    CASE(UR_RESULT_ERROR_INVALID_BUFFER_SIZE, PI_ERROR_INVALID_BUFFER_SIZE)
    CASE(UR_RESULT_ERROR_INVALID_MEM_OBJECT, PI_ERROR_INVALID_MEM_OBJECT)
    CASE(UR_RESULT_ERROR_INVALID_EVENT, PI_ERROR_INVALID_EVENT)
    CASE(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
         PI_ERROR_INVALID_EVENT_WAIT_LIST)
    CASE(UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET,
         PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET)
    CASE(UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE,
         PI_ERROR_INVALID_WORK_GROUP_SIZE)
    CASE(UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE,
         PI_ERROR_COMPILER_NOT_AVAILABLE)
    CASE(UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE,
         PI_ERROR_PROFILING_INFO_NOT_AVAILABLE)
    CASE(UR_RESULT_ERROR_DEVICE_NOT_FOUND, PI_ERROR_DEVICE_NOT_FOUND)
    CASE(UR_RESULT_ERROR_INVALID_DEVICE, PI_ERROR_INVALID_DEVICE)
    CASE(UR_RESULT_ERROR_DEVICE_LOST, PI_ERROR_DEVICE_NOT_AVAILABLE)
    // UR_RESULT_ERROR_DEVICE_REQUIRES_RESET
    // UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE
    CASE(UR_RESULT_ERROR_DEVICE_PARTITION_FAILED,
         PI_ERROR_DEVICE_PARTITION_FAILED)
    CASE(UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT,
         PI_ERROR_INVALID_DEVICE_PARTITION_COUNT)
    CASE(UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE,
         PI_ERROR_INVALID_WORK_ITEM_SIZE)
    CASE(UR_RESULT_ERROR_INVALID_WORK_DIMENSION,
         PI_ERROR_INVALID_WORK_DIMENSION)
    CASE(UR_RESULT_ERROR_INVALID_KERNEL_ARGS, PI_ERROR_INVALID_KERNEL_ARGS)
    CASE(UR_RESULT_ERROR_INVALID_KERNEL, PI_ERROR_INVALID_KERNEL)
    CASE(UR_RESULT_ERROR_INVALID_KERNEL_NAME, PI_ERROR_INVALID_KERNEL_NAME)
    CASE(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
         PI_ERROR_INVALID_ARG_INDEX)
    CASE(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE,
         PI_ERROR_INVALID_ARG_SIZE)
    // UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE
    CASE(UR_RESULT_ERROR_INVALID_IMAGE_SIZE, PI_ERROR_INVALID_IMAGE_SIZE)
    CASE(UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR,
         PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    CASE(UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED,
         PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED)
    CASE(UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE,
         PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE)
    CASE(UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE,
         PI_ERROR_INVALID_PROGRAM_EXECUTABLE)
    CASE(UR_RESULT_ERROR_UNINITIALIZED, PI_ERROR_UNINITIALIZED)
    CASE(UR_RESULT_ERROR_OUT_OF_HOST_MEMORY, PI_ERROR_OUT_OF_HOST_MEMORY)
    CASE(UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, PI_ERROR_OUT_OF_RESOURCES)
    CASE(UR_RESULT_ERROR_OUT_OF_RESOURCES, PI_ERROR_OUT_OF_RESOURCES)
    CASE(UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE, PI_ERROR_BUILD_PROGRAM_FAILURE)
    CASE(UR_RESULT_ERROR_PROGRAM_LINK_FAILURE, PI_ERROR_LINK_PROGRAM_FAILURE)
    // UR_RESULT_ERROR_UNSUPPORTED_VERSION
    // UR_RESULT_ERROR_UNSUPPORTED_FEATURE
    CASE(UR_RESULT_ERROR_INVALID_ARGUMENT, PI_ERROR_INVALID_ARG_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_NULL_HANDLE, PI_ERROR_INVALID_VALUE)
    // UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE
    CASE(UR_RESULT_ERROR_INVALID_NULL_POINTER, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_SIZE, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_UNSUPPORTED_SIZE, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_ENUMERATION, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT,
         PI_ERROR_IMAGE_FORMAT_MISMATCH)
    CASE(UR_RESULT_ERROR_INVALID_NATIVE_BINARY, PI_ERROR_INVALID_BINARY)
    CASE(UR_RESULT_ERROR_INVALID_GLOBAL_NAME, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_FUNCTION_NAME, PI_ERROR_INVALID_VALUE)
    CASE(UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION,
         PI_ERROR_INVALID_WORK_GROUP_SIZE)
    CASE(UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION,
         PI_ERROR_INVALID_GLOBAL_WORK_SIZE)
    // UR_RESULT_ERROR_PROGRAM_UNLINKED
    // UR_RESULT_ERROR_OVERLAPPING_REGIONS
    CASE(UR_RESULT_ERROR_INVALID_HOST_PTR, PI_ERROR_INVALID_HOST_PTR)
    // UR_RESULT_ERROR_INVALID_USM_SIZE
    CASE(UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE,
         PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE)
    CASE(UR_RESULT_ERROR_ADAPTER_SPECIFIC, PI_ERROR_PLUGIN_SPECIFIC_ERROR)

#undef CASE
  default:
    return PI_ERROR_UNKNOWN;
  }
}

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
pi_result hip_piPluginGetLastError(char **message) {
  *message = &ErrorMessage[0];
  return ErrorMessageCode;
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return empty string for hip.
// TODO: Determine correct string to be passed.
pi_result hip_piPluginGetBackendOption(pi_platform, const char *frontend_option,
                                       const char **backend_option) {
  using namespace std::literals;
  if (frontend_option == nullptr)
    return PI_ERROR_INVALID_VALUE;
  if (frontend_option == "-O0"sv || frontend_option == "-O1"sv ||
      frontend_option == "-O2"sv || frontend_option == "-O3"sv ||
      frontend_option == ""sv) {
    *backend_option = "";
    return PI_SUCCESS;
  }
  return PI_ERROR_INVALID_VALUE;
}

/// Converts HIP error into PI error codes, and outputs error information
/// to stderr.
/// If PI_HIP_ABORT env variable is defined, it aborts directly instead of
/// throwing the error. This is intended for debugging purposes.
/// \return PI_SUCCESS if \param result was hipSuccess.
/// \throw pi_error exception (integer) if input was not success.
///
pi_result check_error(hipError_t result, const char *function, int line,
                      const char *file) {
  if (result == hipSuccess) {
    return PI_SUCCESS;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    const char *errorString = nullptr;
    const char *errorName = nullptr;
    errorName = hipGetErrorName(result);
    errorString = hipGetErrorString(result);
    std::stringstream ss;
    ss << "\nPI HIP ERROR:"
       << "\n\tValue:           " << result
       << "\n\tName:            " << errorName
       << "\n\tDescription:     " << errorString
       << "\n\tFunction:        " << function << "\n\tSource Location: " << file
       << ":" << line << "\n"
       << std::endl;
    std::cerr << ss.str();
  }

  if (std::getenv("PI_HIP_ABORT") != nullptr) {
    std::abort();
  }

  throw map_error(result);
}

/// \cond NODOXY
#define PI_CHECK_ERROR(result) check_error(result, __func__, __LINE__, __FILE__)

/// \cond NODOXY
template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return PI_ERROR_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    (void)value_size;
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), std::move(assignment));
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {

  auto assignment = [](void *param_value, T *value, size_t value_size) {
    memcpy(param_value, static_cast<const void *>(value), value_size);
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), std::move(assignment));
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

/// \endcond

void simpleGuessLocalWorkSize(size_t *threadsPerBlock,
                              const size_t *global_work_size,
                              const size_t maxThreadsPerBlock[3],
                              [[maybe_unused]] pi_kernel kernel) {
  assert(threadsPerBlock != nullptr);
  assert(global_work_size != nullptr);
  assert(kernel != nullptr);
  // int recommendedBlockSize, minGrid;

  // PI_CHECK_ERROR(hipOccupancyMaxPotentialBlockSize(
  //    &minGrid, &recommendedBlockSize, kernel->get(),
  //    0, 0));

  //(void)minGrid; // Not used, avoid warnings

  threadsPerBlock[0] = std::min(maxThreadsPerBlock[0], global_work_size[0]);

  // Find a local work group size that is a divisor of the global
  // work group size to produce uniform work groups.
  while (0u != (global_work_size[0] % threadsPerBlock[0])) {
    --threadsPerBlock[0];
  }
}

pi_result enqueueEventsWait(pi_queue command_queue, hipStream_t stream,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list) {
  if (!event_wait_list) {
    return PI_SUCCESS;
  }
  try {
    ScopedContext active(command_queue->get_context());

    auto result = map_ur_error(forLatestEvents(
        reinterpret_cast<const ur_event_handle_t *>(event_wait_list),
        num_events_in_wait_list,
        [stream](ur_event_handle_t event) -> ur_result_t {
          if (event->get_stream() == stream) {
            return UR_RESULT_SUCCESS;
          } else {
            return UR_CHECK_ERROR(hipStreamWaitEvent(stream, event->get(), 0));
          }
        }));

    if (result != PI_SUCCESS) {
      return result;
    }
    return PI_SUCCESS;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

} // anonymous namespace

/// ------ Error handling, matching OpenCL plugin semantics.
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
namespace pi {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message) {
  std::cerr << "pi_die: " << Message << std::endl;
  std::terminate();
}

// Reports error messages
void hipPrint(const char *Message) {
  std::cerr << "pi_print: " << Message << std::endl;
}

void assertion(bool Condition, const char *Message) {
  if (!Condition)
    die(Message);
}

} // namespace pi
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

//--------------
// PI object implementation

extern "C" {

// Required in a number of functions, so forward declare here
pi_result hip_piEnqueueEventsWait(pi_queue command_queue,
                                  pi_uint32 num_events_in_wait_list,
                                  const pi_event *event_wait_list,
                                  pi_event *event);
pi_result hip_piEnqueueEventsWaitWithBarrier(pi_queue command_queue,
                                             pi_uint32 num_events_in_wait_list,
                                             const pi_event *event_wait_list,
                                             pi_event *event);
} // extern "C"

/// \endcond

// makes all future work submitted to queue wait for all work captured in event.
pi_result enqueueEventWait(pi_queue queue, pi_event event) {
  // for native events, the hipStreamWaitEvent call is used.
  // This makes all future work submitted to stream wait for all
  // work captured in event.
  queue->for_each_stream([e = event->get()](hipStream_t s) {
    PI_CHECK_ERROR(hipStreamWaitEvent(s, e, 0));
  });
  return PI_SUCCESS;
}

//-- PI API implementation
extern "C" {

/// \return If available, the first binary that is PTX
///
pi_result hip_piextDeviceSelectBinary(pi_device device,
                                      pi_device_binary *binaries,
                                      pi_uint32 num_binaries,
                                      pi_uint32 *selected_binary) {
  (void)device;
  if (!binaries) {
    sycl::detail::pi::die("No list of device images provided");
  }
  if (num_binaries < 1) {
    sycl::detail::pi::die("No binary images in the list");
  }

  // Look for an image for the HIP target, and return the first one that is
  // found
#if defined(__HIP_PLATFORM_AMD__)
  const char *binary_type = __SYCL_PI_DEVICE_BINARY_TARGET_AMDGCN;
#elif defined(__HIP_PLATFORM_NVIDIA__)
  const char *binary_type = __SYCL_PI_DEVICE_BINARY_TARGET_NVPTX64;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

  for (pi_uint32 i = 0; i < num_binaries; i++) {
    if (strcmp(binaries[i]->DeviceTargetSpec, binary_type) == 0) {
      *selected_binary = i;
      return PI_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return PI_ERROR_INVALID_BINARY;
}

pi_result hip_piextGetDeviceFunctionPointer([[maybe_unused]] pi_device device,
                                            pi_program program,
                                            const char *func_name,
                                            pi_uint64 *func_pointer_ret) {
  // Check if device passed is the same the device bound to the context
  assert(device == program->get_context()->get_device());
  assert(func_pointer_ret != nullptr);

  hipFunction_t func;
  hipError_t ret = hipModuleGetFunction(&func, program->get(), func_name);
  *func_pointer_ret = reinterpret_cast<pi_uint64>(func);
  pi_result retError = PI_SUCCESS;

  if (ret != hipSuccess && ret != hipErrorNotFound)
    retError = PI_CHECK_ERROR(ret);
  if (ret == hipErrorNotFound) {
    *func_pointer_ret = 0;
    retError = PI_ERROR_INVALID_KERNEL_NAME;
  }

  return retError;
}

pi_result hip_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                                      pi_bool blocking_write, size_t offset,
                                      size_t size, void *ptr,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);
  pi_result retErr = PI_SUCCESS;
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();
    retErr = enqueueEventsWait(command_queue, hipStream,
                               num_events_in_wait_list, event_wait_list);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_WRITE, command_queue, hipStream));
      retImplEv->start();
    }

    retErr = PI_CHECK_ERROR(
        hipMemcpyHtoDAsync(buffer->mem_.buffer_mem_.get_with_offset(offset),
                           ptr, size, hipStream));

    if (event) {
      retErr = map_ur_error(retImplEv->record());
    }

    if (blocking_write) {
      retErr = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (event) {
      *event = retImplEv.release();
    }
  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piEnqueueMemBufferRead(pi_queue command_queue, pi_mem buffer,
                                     pi_bool blocking_read, size_t offset,
                                     size_t size, void *ptr,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);
  pi_result retErr = PI_SUCCESS;
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();
    retErr = enqueueEventsWait(command_queue, hipStream,
                               num_events_in_wait_list, event_wait_list);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_READ, command_queue, hipStream));
      retImplEv->start();
    }

    retErr = PI_CHECK_ERROR(hipMemcpyDtoHAsync(
        ptr, buffer->mem_.buffer_mem_.get_with_offset(offset), size,
        hipStream));

    if (event) {
      retErr = map_ur_error(retImplEv->record());
    }

    if (blocking_read) {
      retErr = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (event) {
      *event = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                                      const pi_mem *arg_value) {

  assert(kernel != nullptr);
  assert(arg_value != nullptr);

  // Below sets kernel arg when zero-sized buffers are handled.
  // In such case the corresponding memory is null.
  if (*arg_value == nullptr) {
    kernel->set_kernel_arg(arg_index, 0, nullptr);
    return PI_SUCCESS;
  }

  pi_result retErr = PI_SUCCESS;
  try {
    pi_mem arg_mem = *arg_value;

    if (arg_mem->mem_type_ == _pi_mem::mem_type::surface) {
      auto array = arg_mem->mem_.surface_mem_.get_array();
      hipArray_Format Format;
      size_t NumChannels;
      getArrayDesc(array, Format, NumChannels);
      if (Format != HIP_AD_FORMAT_UNSIGNED_INT32 &&
          Format != HIP_AD_FORMAT_SIGNED_INT32 &&
          Format != HIP_AD_FORMAT_HALF && Format != HIP_AD_FORMAT_FLOAT) {
        sycl::detail::pi::die(
            "PI HIP kernels only support images with channel types int32, "
            "uint32, float, and half.");
      }
      hipSurfaceObject_t hipSurf = arg_mem->mem_.surface_mem_.get_surface();
      kernel->set_kernel_arg(arg_index, sizeof(hipSurf), (void *)&hipSurf);
    } else

    {
      void *hipPtr = arg_mem->mem_.buffer_mem_.get_void();
      kernel->set_kernel_arg(arg_index, sizeof(void *), (void *)&hipPtr);
    }
  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piextKernelSetArgSampler(pi_kernel kernel, pi_uint32 arg_index,
                                       const pi_sampler *arg_value) {

  assert(kernel != nullptr);
  assert(arg_value != nullptr);

  pi_result retErr = PI_SUCCESS;
  try {
    pi_uint32 samplerProps = (*arg_value)->props_;
    kernel->set_kernel_arg(arg_index, sizeof(pi_uint32), (void *)&samplerProps);
  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piEnqueueKernelLaunch(
    pi_queue command_queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  // Preconditions
  assert(command_queue != nullptr);
  assert(command_queue->get_context() == kernel->get_context());
  assert(kernel != nullptr);
  assert(global_work_offset != nullptr);
  assert(work_dim > 0);
  assert(work_dim < 4);

  if (*global_work_size == 0) {
    return hip_piEnqueueEventsWaitWithBarrier(
        command_queue, num_events_in_wait_list, event_wait_list, event);
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t threadsPerBlock[3] = {32u, 1u, 1u};
  size_t maxWorkGroupSize = 0u;
  size_t maxThreadsPerBlock[3] = {};
  bool providedLocalWorkGroupSize = (local_work_size != nullptr);

  {
    pi_result retError = pi2ur::piDeviceGetInfo(
        reinterpret_cast<pi_device>(command_queue->device_),
        PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES, sizeof(maxThreadsPerBlock),
        maxThreadsPerBlock, nullptr);
    assert(retError == PI_SUCCESS);
    (void)retError;

    retError = pi2ur::piDeviceGetInfo(
        reinterpret_cast<pi_device>(command_queue->device_),
        PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize),
        &maxWorkGroupSize, nullptr);
    assert(retError == PI_SUCCESS);
    // The maxWorkGroupsSize = 1024 for AMD GPU
    // The maxThreadsPerBlock = {1024, 1024, 1024}

    if (providedLocalWorkGroupSize) {
      auto isValid = [&](int dim) {
        if (local_work_size[dim] > maxThreadsPerBlock[dim])
          return PI_ERROR_INVALID_WORK_GROUP_SIZE;
        // Checks that local work sizes are a divisor of the global work sizes
        // which includes that the local work sizes are neither larger than the
        // global work sizes and not 0.
        if (0u == local_work_size[dim])
          return PI_ERROR_INVALID_WORK_GROUP_SIZE;
        if (0u != (global_work_size[dim] % local_work_size[dim]))
          return PI_ERROR_INVALID_WORK_GROUP_SIZE;
        threadsPerBlock[dim] = local_work_size[dim];
        return PI_SUCCESS;
      };

      for (size_t dim = 0; dim < work_dim; dim++) {
        auto err = isValid(dim);
        if (err != PI_SUCCESS)
          return err;
      }
    } else {
      simpleGuessLocalWorkSize(threadsPerBlock, global_work_size,
                               maxThreadsPerBlock, kernel);
    }
  }

  if (maxWorkGroupSize <
      size_t(threadsPerBlock[0] * threadsPerBlock[1] * threadsPerBlock[2])) {
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  size_t blocksPerGrid[3] = {1u, 1u, 1u};

  for (size_t i = 0; i < work_dim; i++) {
    blocksPerGrid[i] =
        (global_work_size[i] + threadsPerBlock[i] - 1) / threadsPerBlock[i];
  }

  pi_result retError = PI_SUCCESS;
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    pi_uint32 stream_token;
    _pi_stream_guard guard;
    hipStream_t hipStream = command_queue->get_next_compute_stream(
        num_events_in_wait_list,
        reinterpret_cast<const ur_event_handle_t *>(event_wait_list), guard,
        &stream_token);
    hipFunction_t hipFunc = kernel->get();

    retError = enqueueEventsWait(command_queue, hipStream,
                                 num_events_in_wait_list, event_wait_list);

    // Set the implicit global offset parameter if kernel has offset variant
    if (kernel->get_with_offset_parameter()) {
      std::uint32_t hip_implicit_offset[3] = {0, 0, 0};
      if (global_work_offset) {
        for (size_t i = 0; i < work_dim; i++) {
          hip_implicit_offset[i] =
              static_cast<std::uint32_t>(global_work_offset[i]);
          if (global_work_offset[i] != 0) {
            hipFunc = kernel->get_with_offset_parameter();
          }
        }
      }
      kernel->set_implicit_offset_arg(sizeof(hip_implicit_offset),
                                      hip_implicit_offset);
    }

    auto argIndices = kernel->get_arg_indices();

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_TYPE_NDRANGE_KERNEL, command_queue,
                                 hipStream, stream_token));
      retImplEv->start();
    }

    // Set local mem max size if env var is present
    static const char *local_mem_sz_ptr =
        std::getenv("SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE");

    if (local_mem_sz_ptr) {
      int device_max_local_mem = 0;
      retError = PI_CHECK_ERROR(hipDeviceGetAttribute(
          &device_max_local_mem, hipDeviceAttributeMaxSharedMemoryPerBlock,
          command_queue->get_device()->get()));

      static const int env_val = std::atoi(local_mem_sz_ptr);
      if (env_val <= 0 || env_val > device_max_local_mem) {
        setErrorMessage("Invalid value specified for "
                        "SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE",
                        PI_ERROR_PLUGIN_SPECIFIC_ERROR);
        return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
      }
      retError = PI_CHECK_ERROR(hipFuncSetAttribute(
          hipFunc, hipFuncAttributeMaxDynamicSharedMemorySize, env_val));
    }

    retError = PI_CHECK_ERROR(hipModuleLaunchKernel(
        hipFunc, blocksPerGrid[0], blocksPerGrid[1], blocksPerGrid[2],
        threadsPerBlock[0], threadsPerBlock[1], threadsPerBlock[2],
        kernel->get_local_size(), hipStream, argIndices.data(), nullptr));

    kernel->clear_local_size();

    if (event) {
      retError = map_ur_error(retImplEv->record());
      *event = retImplEv.release();
    }
  } catch (pi_result err) {
    retError = err;
  }
  return retError;
}

/// \TODO Not implemented
pi_result
hip_piEnqueueNativeKernel(pi_queue queue, void (*user_func)(void *), void *args,
                          size_t cb_args, pi_uint32 num_mem_objects,
                          const pi_mem *mem_list, const void **args_mem_loc,
                          pi_uint32 num_events_in_wait_list,
                          const pi_event *event_wait_list, pi_event *event) {
  (void)queue;
  (void)user_func;
  (void)args;
  (void)cb_args;
  (void)num_mem_objects;
  (void)mem_list;
  (void)args_mem_loc;
  (void)num_events_in_wait_list;
  (void)event_wait_list;
  (void)event;

  sycl::detail::pi::die("Not implemented in HIP backend");
  return {};
}

/// Enqueues a wait on the given queue for all events.
/// See \ref enqueueEventWait
///
/// Currently queues are represented by a single in-order stream, therefore
/// every command is an implicit barrier and so hip_piEnqueueEventsWait has the
/// same behavior as hip_piEnqueueEventsWaitWithBarrier. So
/// hip_piEnqueueEventsWait can just call hip_piEnqueueEventsWaitWithBarrier.
pi_result hip_piEnqueueEventsWait(pi_queue command_queue,
                                  pi_uint32 num_events_in_wait_list,
                                  const pi_event *event_wait_list,
                                  pi_event *event) {
  return hip_piEnqueueEventsWaitWithBarrier(
      command_queue, num_events_in_wait_list, event_wait_list, event);
}

/// Enqueues a wait on the given queue for all specified events.
/// See \ref enqueueEventWaitWithBarrier
///
/// If the events list is empty, the enqueued wait will wait on all previous
/// events in the queue.
pi_result hip_piEnqueueEventsWaitWithBarrier(pi_queue command_queue,
                                             pi_uint32 num_events_in_wait_list,
                                             const pi_event *event_wait_list,
                                             pi_event *event) {
  if (!command_queue) {
    return PI_ERROR_INVALID_QUEUE;
  }

  pi_result result;

  try {
    ScopedContext active(command_queue->get_context());
    pi_uint32 stream_token;
    _pi_stream_guard guard;
    hipStream_t hipStream = command_queue->get_next_compute_stream(
        num_events_in_wait_list,
        reinterpret_cast<const ur_event_handle_t *>(event_wait_list), guard,
        &stream_token);
    {
      std::lock_guard<std::mutex> guard(command_queue->barrier_mutex_);
      if (command_queue->barrier_event_ == nullptr) {
        PI_CHECK_ERROR(hipEventCreate(&command_queue->barrier_event_));
      }
      if (num_events_in_wait_list == 0) { //  wait on all work
        if (command_queue->barrier_tmp_event_ == nullptr) {
          PI_CHECK_ERROR(hipEventCreate(&command_queue->barrier_tmp_event_));
        }
        command_queue->sync_streams(
            [hipStream,
             tmp_event = command_queue->barrier_tmp_event_](hipStream_t s) {
              if (hipStream != s) {
                PI_CHECK_ERROR(hipEventRecord(tmp_event, s));
                PI_CHECK_ERROR(hipStreamWaitEvent(hipStream, tmp_event, 0));
              }
            });
      } else { // wait just on given events
        forLatestEvents(
            reinterpret_cast<const ur_event_handle_t *>(event_wait_list),
            num_events_in_wait_list,
            [hipStream](ur_event_handle_t event) -> ur_result_t {
              if (event->get_queue()->has_been_synchronized(
                      event->get_compute_stream_token())) {
                return UR_RESULT_SUCCESS;
              } else {
                return UR_CHECK_ERROR(
                    hipStreamWaitEvent(hipStream, event->get(), 0));
              }
            });
      }

      result = PI_CHECK_ERROR(
          hipEventRecord(command_queue->barrier_event_, hipStream));
      for (unsigned int i = 0;
           i < command_queue->compute_applied_barrier_.size(); i++) {
        command_queue->compute_applied_barrier_[i] = false;
      }
      for (unsigned int i = 0;
           i < command_queue->transfer_applied_barrier_.size(); i++) {
        command_queue->transfer_applied_barrier_[i] = false;
      }
    }
    if (result != PI_SUCCESS) {
      return result;
    }

    if (event) {
      *event = _pi_event::make_native(PI_COMMAND_TYPE_MARKER, command_queue,
                                      hipStream, stream_token);
      (*event)->start();
      (*event)->record();
    }

    return PI_SUCCESS;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

/// General 3D memory copy operation.
/// This function requires the corresponding HIP context to be at the top of
/// the context stack
/// If the source and/or destination is on the device, src_ptr and/or dst_ptr
/// must be a pointer to a hipDevPtr
static pi_result commonEnqueueMemBufferCopyRect(
    hipStream_t hip_stream, pi_buff_rect_region region, const void *src_ptr,
    const hipMemoryType src_type, pi_buff_rect_offset src_offset,
    size_t src_row_pitch, size_t src_slice_pitch, void *dst_ptr,
    const hipMemoryType dst_type, pi_buff_rect_offset dst_offset,
    size_t dst_row_pitch, size_t dst_slice_pitch) {

  assert(region != nullptr);
  assert(src_offset != nullptr);
  assert(dst_offset != nullptr);

  assert(src_type == hipMemoryTypeDevice || src_type == hipMemoryTypeHost);
  assert(dst_type == hipMemoryTypeDevice || dst_type == hipMemoryTypeHost);

  src_row_pitch = (!src_row_pitch) ? region->width_bytes : src_row_pitch;
  src_slice_pitch = (!src_slice_pitch) ? (region->height_scalar * src_row_pitch)
                                       : src_slice_pitch;
  dst_row_pitch = (!dst_row_pitch) ? region->width_bytes : dst_row_pitch;
  dst_slice_pitch = (!dst_slice_pitch) ? (region->height_scalar * dst_row_pitch)
                                       : dst_slice_pitch;

  HIP_MEMCPY3D params;

  params.WidthInBytes = region->width_bytes;
  params.Height = region->height_scalar;
  params.Depth = region->depth_scalar;

  params.srcMemoryType = src_type;
  params.srcDevice = src_type == hipMemoryTypeDevice
                         ? *static_cast<const hipDeviceptr_t *>(src_ptr)
                         : 0;
  params.srcHost = src_type == hipMemoryTypeHost ? src_ptr : nullptr;
  params.srcXInBytes = src_offset->x_bytes;
  params.srcY = src_offset->y_scalar;
  params.srcZ = src_offset->z_scalar;
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstMemoryType = dst_type;
  params.dstDevice = dst_type == hipMemoryTypeDevice
                         ? *reinterpret_cast<hipDeviceptr_t *>(dst_ptr)
                         : 0;
  params.dstHost = dst_type == hipMemoryTypeHost ? dst_ptr : nullptr;
  params.dstXInBytes = dst_offset->x_bytes;
  params.dstY = dst_offset->y_scalar;
  params.dstZ = dst_offset->z_scalar;
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;

  return PI_CHECK_ERROR(hipDrvMemcpy3DAsync(&params, hip_stream));

  return PI_SUCCESS;
}

pi_result hip_piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  void *devPtr = buffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();

    retErr = enqueueEventsWait(command_queue, hipStream,
                               num_events_in_wait_list, event_wait_list);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT, command_queue, hipStream));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, &devPtr, hipMemoryTypeDevice, buffer_offset,
        buffer_row_pitch, buffer_slice_pitch, ptr, hipMemoryTypeHost,
        host_offset, host_row_pitch, host_slice_pitch);

    if (event) {
      retErr = map_ur_error(retImplEv->record());
    }

    if (blocking_read) {
      retErr = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (event) {
      *event = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  void *devPtr = buffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();
    retErr = enqueueEventsWait(command_queue, hipStream,
                               num_events_in_wait_list, event_wait_list);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_WRITE_RECT, command_queue, hipStream));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, ptr, hipMemoryTypeHost, host_offset, host_row_pitch,
        host_slice_pitch, &devPtr, hipMemoryTypeDevice, buffer_offset,
        buffer_row_pitch, buffer_slice_pitch);

    if (event) {
      retErr = map_ur_error(retImplEv->record());
    }

    if (blocking_write) {
      retErr = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (event) {
      *event = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                                     pi_mem dst_buffer, size_t src_offset,
                                     size_t dst_offset, size_t size,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  if (!command_queue) {
    return PI_ERROR_INVALID_QUEUE;
  }

  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    pi_result result;
    auto stream = command_queue->get_next_transfer_stream();

    if (event_wait_list) {
      result = enqueueEventsWait(command_queue, stream, num_events_in_wait_list,
                                 event_wait_list);
    }

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_COPY, command_queue, stream));
      result = map_ur_error(retImplEv->start());
    }

    auto src = src_buffer->mem_.buffer_mem_.get_with_offset(src_offset);
    auto dst = dst_buffer->mem_.buffer_mem_.get_with_offset(dst_offset);

    result = PI_CHECK_ERROR(hipMemcpyDtoDAsync(dst, src, size, stream));

    if (event) {
      result = map_ur_error(retImplEv->record());
      *event = retImplEv.release();
    }

    return result;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

pi_result hip_piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    pi_buff_rect_offset src_origin, pi_buff_rect_offset dst_origin,
    pi_buff_rect_region region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t dst_row_pitch, size_t dst_slice_pitch,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {

  assert(src_buffer != nullptr);
  assert(dst_buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  void *srcPtr = src_buffer->mem_.buffer_mem_.get_void();
  void *dstPtr = dst_buffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();
    retErr = enqueueEventsWait(command_queue, hipStream,
                               num_events_in_wait_list, event_wait_list);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, command_queue, hipStream));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, &srcPtr, hipMemoryTypeDevice, src_origin,
        src_row_pitch, src_slice_pitch, &dstPtr, hipMemoryTypeDevice,
        dst_origin, dst_row_pitch, dst_slice_pitch);

    if (event) {
      retImplEv->record();
      *event = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result hip_piEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
                                     const void *pattern, size_t pattern_size,
                                     size_t offset, size_t size,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  assert(command_queue != nullptr);

  auto args_are_multiples_of_pattern_size =
      (offset % pattern_size == 0) || (size % pattern_size == 0);

  auto pattern_is_valid = (pattern != nullptr);

  auto pattern_size_is_valid =
      ((pattern_size & (pattern_size - 1)) == 0) && // is power of two
      (pattern_size > 0) && (pattern_size <= 128);  // falls within valid range

  assert(args_are_multiples_of_pattern_size && pattern_is_valid &&
         pattern_size_is_valid);
  (void)args_are_multiples_of_pattern_size;
  (void)pattern_is_valid;
  (void)pattern_size_is_valid;

  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    auto stream = command_queue->get_next_transfer_stream();
    pi_result result;
    if (event_wait_list) {
      result = enqueueEventsWait(command_queue, stream, num_events_in_wait_list,
                                 event_wait_list);
    }

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_FILL, command_queue, stream));
      result = map_ur_error(retImplEv->start());
    }

    auto dstDevice = buffer->mem_.buffer_mem_.get_with_offset(offset);
    auto N = size / pattern_size;

    // pattern size in bytes
    switch (pattern_size) {
    case 1: {
      auto value = *static_cast<const uint8_t *>(pattern);
      result = PI_CHECK_ERROR(hipMemsetD8Async(dstDevice, value, N, stream));
      break;
    }
    case 2: {
      auto value = *static_cast<const uint16_t *>(pattern);
      result = PI_CHECK_ERROR(hipMemsetD16Async(dstDevice, value, N, stream));
      break;
    }
    case 4: {
      auto value = *static_cast<const uint32_t *>(pattern);
      result = PI_CHECK_ERROR(hipMemsetD32Async(dstDevice, value, N, stream));
      break;
    }

    default: {
      // HIP has no memset functions that allow setting values more than 4
      // bytes. PI API lets you pass an arbitrary "pattern" to the buffer
      // fill, which can be more than 4 bytes. We must break up the pattern
      // into 1 byte values, and set the buffer using multiple strided calls.
      // The first 4 patterns are set using hipMemsetD32Async then all
      // subsequent 1 byte patterns are set using hipMemset2DAsync which is
      // called for each pattern.

      // Calculate the number of patterns, stride, number of times the pattern
      // needs to be applied, and the number of times the first 32 bit pattern
      // needs to be applied.
      auto number_of_steps = pattern_size / sizeof(uint8_t);
      auto pitch = number_of_steps * sizeof(uint8_t);
      auto height = size / number_of_steps;
      auto count_32 = size / sizeof(uint32_t);

      // Get 4-byte chunk of the pattern and call hipMemsetD32Async
      auto value = *(static_cast<const uint32_t *>(pattern));
      result =
          PI_CHECK_ERROR(hipMemsetD32Async(dstDevice, value, count_32, stream));
      for (auto step = 4u; step < number_of_steps; ++step) {
        // take 1 byte of the pattern
        value = *(static_cast<const uint8_t *>(pattern) + step);

        // offset the pointer to the part of the buffer we want to write to
        auto offset_ptr = reinterpret_cast<void *>(
            reinterpret_cast<uint8_t *>(dstDevice) + (step * sizeof(uint8_t)));

        // set all of the pattern chunks
        result = PI_CHECK_ERROR(hipMemset2DAsync(
            offset_ptr, pitch, value, sizeof(uint8_t), height, stream));
      }
      break;
    }
    }

    if (event) {
      result = map_ur_error(retImplEv->record());
      *event = retImplEv.release();
    }

    return result;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

static size_t imageElementByteSize(hipArray_Format array_format) {
  switch (array_format) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
  case HIP_AD_FORMAT_SIGNED_INT8:
    return 1;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
  case HIP_AD_FORMAT_SIGNED_INT16:
  case HIP_AD_FORMAT_HALF:
    return 2;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
  case HIP_AD_FORMAT_SIGNED_INT32:
  case HIP_AD_FORMAT_FLOAT:
    return 4;
  default:
    return 0;
  }
  sycl::detail::pi::die("Invalid iamge format.");
  return 0;
}

/// General ND memory copy operation for images (where N > 1).
/// This function requires the corresponding HIP context to be at the top of
/// the context stack
/// If the source and/or destination is an array, src_ptr and/or dst_ptr
/// must be a pointer to a hipArray

static pi_result commonEnqueueMemImageNDCopy(
    hipStream_t hip_stream, pi_mem_type img_type, const size_t *region,
    const void *src_ptr, const hipMemoryType src_type, const size_t *src_offset,
    void *dst_ptr, const hipMemoryType dst_type, const size_t *dst_offset) {
  assert(region != nullptr);

  assert(src_type == hipMemoryTypeArray || src_type == hipMemoryTypeHost);
  assert(dst_type == hipMemoryTypeArray || dst_type == hipMemoryTypeHost);

  if (img_type == PI_MEM_TYPE_IMAGE2D) {
    hip_Memcpy2D cpyDesc;
    memset(&cpyDesc, 0, sizeof(cpyDesc));
    cpyDesc.srcMemoryType = src_type;
    if (src_type == hipMemoryTypeArray) {
      cpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(src_ptr));
      cpyDesc.srcXInBytes = src_offset[0];
      cpyDesc.srcY = src_offset[1];
    } else {
      cpyDesc.srcHost = src_ptr;
    }
    cpyDesc.dstMemoryType = dst_type;
    if (dst_type == hipMemoryTypeArray) {
      cpyDesc.dstArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(dst_ptr));
      cpyDesc.dstXInBytes = dst_offset[0];
      cpyDesc.dstY = dst_offset[1];
    } else {
      cpyDesc.dstHost = dst_ptr;
    }
    cpyDesc.WidthInBytes = region[0];
    cpyDesc.Height = region[1];
    return PI_CHECK_ERROR(hipMemcpyParam2DAsync(&cpyDesc, hip_stream));
  }

  if (img_type == PI_MEM_TYPE_IMAGE3D) {

    HIP_MEMCPY3D cpyDesc;
    memset(&cpyDesc, 0, sizeof(cpyDesc));
    cpyDesc.srcMemoryType = src_type;
    if (src_type == hipMemoryTypeArray) {
      cpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(src_ptr));
      cpyDesc.srcXInBytes = src_offset[0];
      cpyDesc.srcY = src_offset[1];
      cpyDesc.srcZ = src_offset[2];
    } else {
      cpyDesc.srcHost = src_ptr;
    }
    cpyDesc.dstMemoryType = dst_type;
    if (dst_type == hipMemoryTypeArray) {
      cpyDesc.dstArray = reinterpret_cast<hipCUarray>(dst_ptr);
      cpyDesc.dstXInBytes = dst_offset[0];
      cpyDesc.dstY = dst_offset[1];
      cpyDesc.dstZ = dst_offset[2];
    } else {
      cpyDesc.dstHost = dst_ptr;
    }
    cpyDesc.WidthInBytes = region[0];
    cpyDesc.Height = region[1];
    cpyDesc.Depth = region[2];
    return PI_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpyDesc, hip_stream));
    return PI_ERROR_UNKNOWN;
  }

  return PI_ERROR_INVALID_VALUE;
}

// TODO(ur) - this is just a workaround until we port Enqueue
static std::unordered_map<ur_mem_type_t, pi_mem_type> UrToPiMemTypeMap = {
    {UR_MEM_TYPE_BUFFER, PI_MEM_TYPE_BUFFER},
    {UR_MEM_TYPE_IMAGE2D, PI_MEM_TYPE_IMAGE2D},
    {UR_MEM_TYPE_IMAGE3D, PI_MEM_TYPE_IMAGE3D},
    {UR_MEM_TYPE_IMAGE2D_ARRAY, PI_MEM_TYPE_IMAGE2D_ARRAY},
    {UR_MEM_TYPE_IMAGE1D, PI_MEM_TYPE_IMAGE1D},
    {UR_MEM_TYPE_IMAGE1D_ARRAY, PI_MEM_TYPE_IMAGE1D_ARRAY},
    {UR_MEM_TYPE_IMAGE1D_BUFFER, PI_MEM_TYPE_IMAGE1D_BUFFER},
};

pi_result hip_piEnqueueMemImageRead(pi_queue command_queue, pi_mem image,
                                    pi_bool blocking_read, const size_t *origin,
                                    const size_t *region, size_t row_pitch,
                                    size_t slice_pitch, void *ptr,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {
  (void)row_pitch;
  (void)slice_pitch;

  assert(command_queue != nullptr);
  assert(image != nullptr);
  assert(image->mem_type_ == _pi_mem::mem_type::surface);

  pi_result retErr = PI_SUCCESS;

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();

    if (event_wait_list) {
      retErr = enqueueEventsWait(command_queue, hipStream,
                                 num_events_in_wait_list, event_wait_list);
    }

    hipArray *array = image->mem_.surface_mem_.get_array();

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(array, Format, NumChannels);

    int elementByteSize = imageElementByteSize(Format);

    size_t byteOffsetX = origin[0] * elementByteSize * NumChannels;
    size_t bytesToCopy = elementByteSize * NumChannels * region[0];

    // TODO(ur) - this can be removed when porting Enqueue
    auto urImgType = image->mem_.surface_mem_.get_image_type();
    pi_mem_type imgType;
    if (auto search = UrToPiMemTypeMap.find(urImgType);
        search != UrToPiMemTypeMap.end()) {
      imgType = search->second;
    } else {
      return PI_ERROR_UNKNOWN;
    }

    size_t adjustedRegion[3] = {bytesToCopy, region[1], region[2]};
    size_t srcOffset[3] = {byteOffsetX, origin[1], origin[2]};

    retErr = commonEnqueueMemImageNDCopy(hipStream, imgType, adjustedRegion,
                                         array, hipMemoryTypeArray, srcOffset,
                                         ptr, hipMemoryTypeHost, nullptr);

    if (retErr != PI_SUCCESS) {
      return retErr;
    }

    if (event) {
      auto new_event = _pi_event::make_native(PI_COMMAND_TYPE_IMAGE_READ,
                                              command_queue, hipStream);
      new_event->record();
      *event = new_event;
    }

    if (blocking_read) {
      retErr = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
  return retErr;
}

pi_result hip_piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                                     pi_bool blocking_write,
                                     const size_t *origin, const size_t *region,
                                     size_t input_row_pitch,
                                     size_t input_slice_pitch, const void *ptr,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  (void)blocking_write;
  (void)input_row_pitch;
  (void)input_slice_pitch;
  assert(command_queue != nullptr);
  assert(image != nullptr);
  assert(image->mem_type_ == _pi_mem::mem_type::surface);

  pi_result retErr = PI_SUCCESS;

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();

    if (event_wait_list) {
      retErr = enqueueEventsWait(command_queue, hipStream,
                                 num_events_in_wait_list, event_wait_list);
    }

    hipArray *array = image->mem_.surface_mem_.get_array();

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(array, Format, NumChannels);

    int elementByteSize = imageElementByteSize(Format);

    size_t byteOffsetX = origin[0] * elementByteSize * NumChannels;
    size_t bytesToCopy = elementByteSize * NumChannels * region[0];

    // TODO(ur) - this can be removed when porting Enqueue
    auto urImgType = image->mem_.surface_mem_.get_image_type();
    pi_mem_type imgType;
    if (auto search = UrToPiMemTypeMap.find(urImgType);
        search != UrToPiMemTypeMap.end()) {
      imgType = search->second;
    } else {
      return PI_ERROR_UNKNOWN;
    }

    size_t adjustedRegion[3] = {bytesToCopy, region[1], region[2]};
    size_t dstOffset[3] = {byteOffsetX, origin[1], origin[2]};

    retErr = commonEnqueueMemImageNDCopy(hipStream, imgType, adjustedRegion,
                                         ptr, hipMemoryTypeHost, nullptr, array,
                                         hipMemoryTypeArray, dstOffset);

    if (retErr != PI_SUCCESS) {
      return retErr;
    }

    if (event) {
      auto new_event = _pi_event::make_native(PI_COMMAND_TYPE_IMAGE_WRITE,
                                              command_queue, hipStream);
      new_event->record();
      *event = new_event;
    }
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;

  return retErr;
}

pi_result hip_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                                    pi_mem dst_image, const size_t *src_origin,
                                    const size_t *dst_origin,
                                    const size_t *region,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {

  assert(src_image->mem_type_ == _pi_mem::mem_type::surface);
  assert(dst_image->mem_type_ == _pi_mem::mem_type::surface);
  assert(src_image->mem_.surface_mem_.get_image_type() ==
         dst_image->mem_.surface_mem_.get_image_type());

  pi_result retErr = PI_SUCCESS;

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get_next_transfer_stream();
    if (event_wait_list) {
      retErr = enqueueEventsWait(command_queue, hipStream,
                                 num_events_in_wait_list, event_wait_list);
    }

    hipArray *srcArray = src_image->mem_.surface_mem_.get_array();
    hipArray_Format srcFormat;
    size_t srcNumChannels;
    getArrayDesc(srcArray, srcFormat, srcNumChannels);

    hipArray *dstArray = dst_image->mem_.surface_mem_.get_array();
    hipArray_Format dstFormat;
    size_t dstNumChannels;
    getArrayDesc(dstArray, dstFormat, dstNumChannels);

    assert(srcFormat == dstFormat);
    assert(srcNumChannels == dstNumChannels);

    int elementByteSize = imageElementByteSize(srcFormat);

    size_t dstByteOffsetX = dst_origin[0] * elementByteSize * srcNumChannels;
    size_t srcByteOffsetX = src_origin[0] * elementByteSize * dstNumChannels;
    size_t bytesToCopy = elementByteSize * srcNumChannels * region[0];

    // TODO(ur) - this can be removed when porting Enqueue
    auto urImgType = src_image->mem_.surface_mem_.get_image_type();
    pi_mem_type imgType;
    if (auto search = UrToPiMemTypeMap.find(urImgType);
        search != UrToPiMemTypeMap.end()) {
      imgType = search->second;
    } else {
      return PI_ERROR_UNKNOWN;
    }

    size_t adjustedRegion[3] = {bytesToCopy, region[1], region[2]};
    size_t srcOffset[3] = {srcByteOffsetX, src_origin[1], src_origin[2]};
    size_t dstOffset[3] = {dstByteOffsetX, dst_origin[1], dst_origin[2]};

    retErr = commonEnqueueMemImageNDCopy(
        hipStream, imgType, adjustedRegion, srcArray, hipMemoryTypeArray,
        srcOffset, dstArray, hipMemoryTypeArray, dstOffset);

    if (retErr != PI_SUCCESS) {
      return retErr;
    }

    if (event) {
      auto new_event = _pi_event::make_native(PI_COMMAND_TYPE_IMAGE_COPY,
                                              command_queue, hipStream);
      new_event->record();
      *event = new_event;
    }
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
  return retErr;
}

/// \TODO Not implemented in HIP.
pi_result hip_piEnqueueMemImageFill(pi_queue command_queue, pi_mem image,
                                    const void *fill_color,
                                    const size_t *origin, const size_t *region,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event) {
  (void)command_queue;
  (void)image;
  (void)fill_color;
  (void)origin;
  (void)region;
  (void)num_events_in_wait_list;
  (void)event_wait_list;
  (void)event;

  sycl::detail::pi::die("hip_piEnqueueMemImageFill not implemented");
  return {};
}

/// Implements mapping on the host using a BufferRead operation.
/// Mapped pointers are stored in the pi_mem object.
/// If the buffer uses pinned host memory a pointer to that memory is returned
/// and no read operation is done.
///
pi_result hip_piEnqueueMemBufferMap(pi_queue command_queue, pi_mem buffer,
                                    pi_bool blocking_map,
                                    pi_map_flags map_flags, size_t offset,
                                    size_t size,
                                    pi_uint32 num_events_in_wait_list,
                                    const pi_event *event_wait_list,
                                    pi_event *event, void **ret_map) {
  assert(ret_map != nullptr);
  assert(command_queue != nullptr);
  assert(buffer != nullptr);
  assert(buffer->mem_type_ == _pi_mem::mem_type::buffer);

  pi_result ret_err = PI_ERROR_INVALID_OPERATION;
  const bool is_pinned = buffer->mem_.buffer_mem_.allocMode_ ==
                         _pi_mem::mem_::buffer_mem_::alloc_mode::alloc_host_ptr;

  // Currently no support for overlapping regions
  if (buffer->mem_.buffer_mem_.get_map_ptr() != nullptr) {
    return ret_err;
  }

  // Allocate a pointer in the host to store the mapped information
  auto hostPtr = buffer->mem_.buffer_mem_.map_to_ptr(offset, map_flags);
  *ret_map = buffer->mem_.buffer_mem_.get_map_ptr();
  if (hostPtr) {
    ret_err = PI_SUCCESS;
  }

  if (!is_pinned && ((map_flags & PI_MAP_READ) || (map_flags & PI_MAP_WRITE))) {
    // Pinned host memory is already on host so it doesn't need to be read.
    ret_err = hip_piEnqueueMemBufferRead(
        command_queue, buffer, blocking_map, offset, size, hostPtr,
        num_events_in_wait_list, event_wait_list, event);
  } else {
    ScopedContext active(command_queue->get_context());

    if (is_pinned) {
      ret_err = hip_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                        event_wait_list, nullptr);
    }

    if (event) {
      try {
        *event = _pi_event::make_native(
            PI_COMMAND_TYPE_MEM_BUFFER_MAP, command_queue,
            command_queue->get_next_transfer_stream());
        (*event)->start();
        (*event)->record();
      } catch (pi_result error) {
        ret_err = error;
      }
    }
  }

  return ret_err;
}

/// Implements the unmap from the host, using a BufferWrite operation.
/// Requires the mapped pointer to be already registered in the given memobj.
/// If memobj uses pinned host memory, this will not do a write.
///
pi_result hip_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                void *mapped_ptr,
                                pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list,
                                pi_event *event) {
  pi_result ret_err = PI_SUCCESS;

  assert(command_queue != nullptr);
  assert(mapped_ptr != nullptr);
  assert(memobj != nullptr);
  assert(memobj->mem_type_ == _pi_mem::mem_type::buffer);
  assert(memobj->mem_.buffer_mem_.get_map_ptr() != nullptr);
  assert(memobj->mem_.buffer_mem_.get_map_ptr() == mapped_ptr);

  const bool is_pinned = memobj->mem_.buffer_mem_.allocMode_ ==
                         _pi_mem::mem_::buffer_mem_::alloc_mode::alloc_host_ptr;

  if (!is_pinned &&
      ((memobj->mem_.buffer_mem_.get_map_flags() & PI_MAP_WRITE) ||
       (memobj->mem_.buffer_mem_.get_map_flags() &
        PI_MAP_WRITE_INVALIDATE_REGION))) {
    // Pinned host memory is only on host so it doesn't need to be written to.
    ret_err = hip_piEnqueueMemBufferWrite(
        command_queue, memobj, true,
        memobj->mem_.buffer_mem_.get_map_offset(mapped_ptr),
        memobj->mem_.buffer_mem_.get_size(), mapped_ptr,
        num_events_in_wait_list, event_wait_list, event);
  } else {
    ScopedContext active(command_queue->get_context());

    if (is_pinned) {
      ret_err = hip_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                        event_wait_list, nullptr);
    }

    if (event) {
      try {
        *event = _pi_event::make_native(
            PI_COMMAND_TYPE_MEM_BUFFER_UNMAP, command_queue,
            command_queue->get_next_transfer_stream());
        (*event)->start();
        (*event)->record();
      } catch (pi_result error) {
        ret_err = error;
      }
    }
  }

  memobj->mem_.buffer_mem_.unmap(mapped_ptr);
  return ret_err;
}

pi_result hip_piextUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                    size_t count,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {

  assert(queue != nullptr);
  assert(ptr != nullptr);
  pi_result result = PI_SUCCESS;
  std::unique_ptr<_pi_event> event_ptr{nullptr};

  try {
    ScopedContext active(queue->get_context());
    pi_uint32 stream_token;
    _pi_stream_guard guard;
    hipStream_t hipStream = queue->get_next_compute_stream(
        num_events_in_waitlist,
        reinterpret_cast<const ur_event_handle_t *>(events_waitlist), guard,
        &stream_token);
    result = enqueueEventsWait(queue, hipStream, num_events_in_waitlist,
                               events_waitlist);
    if (event) {
      event_ptr = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_FILL, queue, hipStream, stream_token));
      event_ptr->start();
    }
    result = PI_CHECK_ERROR(
        hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                         (unsigned char)value & 0xFF, count, hipStream));
    if (event) {
      result = map_ur_error(event_ptr->record());
      *event = event_ptr.release();
    }
  } catch (pi_result err) {
    result = err;
  }

  return result;
}

pi_result hip_piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                    void *dst_ptr, const void *src_ptr,
                                    size_t size,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  assert(queue != nullptr);
  assert(dst_ptr != nullptr);
  assert(src_ptr != nullptr);
  pi_result result = PI_SUCCESS;

  std::unique_ptr<_pi_event> event_ptr{nullptr};

  try {
    ScopedContext active(queue->get_context());
    hipStream_t hipStream = queue->get_next_transfer_stream();
    result = enqueueEventsWait(queue, hipStream, num_events_in_waitlist,
                               events_waitlist);
    if (event) {
      event_ptr = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_COPY, queue, hipStream));
      event_ptr->start();
    }
    result = PI_CHECK_ERROR(
        hipMemcpyAsync(dst_ptr, src_ptr, size, hipMemcpyDefault, hipStream));
    if (event) {
      result = map_ur_error(event_ptr->record());
    }
    if (blocking) {
      result = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }
    if (event) {
      *event = event_ptr.release();
    }
  } catch (pi_result err) {
    result = err;
  }
  return result;
}

pi_result hip_piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                      size_t size, pi_usm_migration_flags flags,
                                      pi_uint32 num_events_in_waitlist,
                                      const pi_event *events_waitlist,
                                      pi_event *event) {

  // flags is currently unused so fail if set
  if (flags != 0)
    return PI_ERROR_INVALID_VALUE;
  assert(queue != nullptr);
  assert(ptr != nullptr);
  pi_result result = PI_SUCCESS;
  std::unique_ptr<_pi_event> event_ptr{nullptr};

  try {
    ScopedContext active(queue->get_context());
    hipStream_t hipStream = queue->get_next_transfer_stream();
    result = enqueueEventsWait(queue, hipStream, num_events_in_waitlist,
                               events_waitlist);
    if (event) {
      event_ptr = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_COPY, queue, hipStream));
      event_ptr->start();
    }
    result = PI_CHECK_ERROR(hipMemPrefetchAsync(
        ptr, size, queue->get_context()->get_device()->get(), hipStream));
    if (event) {
      result = map_ur_error(event_ptr->record());
      *event = event_ptr.release();
    }
  } catch (pi_result err) {
    result = err;
  }

  return result;
}

/// USM: memadvise API to govern behavior of automatic migration mechanisms
pi_result hip_piextUSMEnqueueMemAdvise(pi_queue queue,
                                       [[maybe_unused]] const void *ptr,
                                       size_t length, pi_mem_advice advice,
                                       pi_event *event) {
  (void)length;
  (void)advice;

  assert(queue != nullptr);
  assert(ptr != nullptr);
  // TODO implement a mapping to hipMemAdvise once the expected behaviour
  // of piextUSMEnqueueMemAdvise is detailed in the USM extension
  return hip_piEnqueueEventsWait(queue, 0, nullptr, event);

  return PI_SUCCESS;
}

// TODO: Implement this. Remember to return true for
//       PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT when it is implemented.
pi_result hip_piextUSMEnqueueFill2D(pi_queue, void *, size_t, size_t,
                                    const void *, size_t, size_t, pi_uint32,
                                    const pi_event *, pi_event *) {
  sycl::detail::pi::die("piextUSMEnqueueFill2D: not implemented");
  return {};
}

// TODO: Implement this. Remember to return true for
//       PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT when it is implemented.
pi_result hip_piextUSMEnqueueMemset2D(pi_queue, void *, size_t, int, size_t,
                                      size_t, pi_uint32, const pi_event *,
                                      pi_event *) {
  sycl::detail::pi::die("hip_piextUSMEnqueueMemset2D: not implemented");
  return {};
}

/// 2D Memcpy API
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
pi_result hip_piextUSMEnqueueMemcpy2D(pi_queue queue, pi_bool blocking,
                                      void *dst_ptr, size_t dst_pitch,
                                      const void *src_ptr, size_t src_pitch,
                                      size_t width, size_t height,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  assert(queue != nullptr);

  pi_result result = PI_SUCCESS;

  try {
    ScopedContext active(queue->get_context());
    hipStream_t hipStream = queue->get_next_transfer_stream();
    result = enqueueEventsWait(queue, hipStream, num_events_in_wait_list,
                               event_wait_list);
    if (event) {
      (*event) = _pi_event::make_native(PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT,
                                        queue, hipStream);
      (*event)->start();
    }

    result = PI_CHECK_ERROR(hipMemcpy2DAsync(dst_ptr, dst_pitch, src_ptr,
                                             src_pitch, width, height,
                                             hipMemcpyDefault, hipStream));

    if (event) {
      (*event)->record();
    }
    if (blocking) {
      result = PI_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }
  } catch (pi_result err) {
    result = err;
  }

  return result;
}

pi_result hip_piextEnqueueDeviceGlobalVariableWrite(
    pi_queue queue, pi_program program, const char *name,
    pi_bool blocking_write, size_t count, size_t offset, const void *src,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {
  (void)queue;
  (void)program;
  (void)name;
  (void)blocking_write;
  (void)count;
  (void)offset;
  (void)src;
  (void)num_events_in_wait_list;
  (void)event_wait_list;
  (void)event;

  sycl::detail::pi::die(
      "hip_piextEnqueueDeviceGlobalVariableWrite not implemented");
  return {};
}

pi_result hip_piextEnqueueDeviceGlobalVariableRead(
    pi_queue queue, pi_program program, const char *name, pi_bool blocking_read,
    size_t count, size_t offset, void *dst, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  (void)queue;
  (void)program;
  (void)name;
  (void)blocking_read;
  (void)count;
  (void)offset;
  (void)dst;
  (void)num_events_in_wait_list;
  (void)event_wait_list;
  (void)event;

  sycl::detail::pi::die(
      "hip_piextEnqueueDeviceGlobalVariableRead not implemented");
}

/// Host Pipes
pi_result hip_piextEnqueueReadHostPipe(pi_queue queue, pi_program program,
                                       const char *pipe_symbol,
                                       pi_bool blocking, void *ptr, size_t size,
                                       pi_uint32 num_events_in_waitlist,
                                       const pi_event *events_waitlist,
                                       pi_event *event) {
  (void)queue;
  (void)program;
  (void)pipe_symbol;
  (void)blocking;
  (void)ptr;
  (void)size;
  (void)num_events_in_waitlist;
  (void)events_waitlist;
  (void)event;

  sycl::detail::pi::die("hip_piextEnqueueReadHostPipe not implemented");
  return {};
}

pi_result hip_piextEnqueueWriteHostPipe(
    pi_queue queue, pi_program program, const char *pipe_symbol,
    pi_bool blocking, void *ptr, size_t size, pi_uint32 num_events_in_waitlist,
    const pi_event *events_waitlist, pi_event *event) {
  (void)queue;
  (void)program;
  (void)pipe_symbol;
  (void)blocking;
  (void)ptr;
  (void)size;
  (void)num_events_in_waitlist;
  (void)events_waitlist;
  (void)event;

  sycl::detail::pi::die("hip_piextEnqueueWriteHostPipe not implemented");
  return {};
}

pi_result hip_piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                      uint64_t *HostTime) {
  if (!DeviceTime && !HostTime)
    return PI_SUCCESS;

  _pi_event::native_type event;

  ScopedContext active(Device->get_context());

  if (DeviceTime) {
    PI_CHECK_ERROR(hipEventCreateWithFlags(&event, hipEventDefault));
    PI_CHECK_ERROR(hipEventRecord(event));
  }
  if (HostTime) {
    using namespace std::chrono;
    *HostTime =
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count();
  }

  if (DeviceTime) {
    PI_CHECK_ERROR(hipEventSynchronize(event));

    float elapsedTime = 0.0f;
    PI_CHECK_ERROR(hipEventElapsedTime(&elapsedTime,
                                       ur_platform_handle_t_::evBase_, event));
    *DeviceTime = (uint64_t)(elapsedTime * (double)1e6);
  }
  return PI_SUCCESS;
}

const char SupportedVersion[] = _PI_HIP_PLUGIN_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // PI interface supports higher version or the same version.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  if (strlen(SupportedVersion) >= PluginVersionSize)
    return PI_ERROR_INVALID_VALUE;
  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

  // Set whole function table to zero to make it easier to detect if
  // functions are not set up below.
  std::memset(&(PluginInit->PiFunctionTable), 0,
              sizeof(PluginInit->PiFunctionTable));

// Forward calls to HIP RT.
#define _PI_CL(pi_api, hip_api)                                                \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&hip_api);

  // Platform
  _PI_CL(piPlatformsGet, pi2ur::piPlatformsGet)
  _PI_CL(piPlatformGetInfo, pi2ur::piPlatformGetInfo)
  // Device
  _PI_CL(piDevicesGet, pi2ur::piDevicesGet)
  _PI_CL(piDeviceGetInfo, pi2ur::piDeviceGetInfo)
  _PI_CL(piDevicePartition, pi2ur::piDevicePartition)
  _PI_CL(piDeviceRetain, pi2ur::piDeviceRetain)
  _PI_CL(piDeviceRelease, pi2ur::piDeviceRelease)
  _PI_CL(piextDeviceSelectBinary, hip_piextDeviceSelectBinary)
  _PI_CL(piextGetDeviceFunctionPointer, hip_piextGetDeviceFunctionPointer)
  _PI_CL(piextDeviceGetNativeHandle, pi2ur::piextDeviceGetNativeHandle)
  _PI_CL(piextDeviceCreateWithNativeHandle,
         pi2ur::piextDeviceCreateWithNativeHandle)
  // Context
  _PI_CL(piextContextSetExtendedDeleter, pi2ur::piextContextSetExtendedDeleter)
  _PI_CL(piContextCreate, pi2ur::piContextCreate)
  _PI_CL(piContextGetInfo, pi2ur::piContextGetInfo)
  _PI_CL(piContextRetain, pi2ur::piContextRetain)
  _PI_CL(piContextRelease, pi2ur::piContextRelease)
  _PI_CL(piextContextGetNativeHandle, pi2ur::piextContextGetNativeHandle)
  _PI_CL(piextContextCreateWithNativeHandle,
         pi2ur::piextContextCreateWithNativeHandle)
  // Queue
  _PI_CL(piQueueCreate, pi2ur::piQueueCreate)
  _PI_CL(piextQueueCreate, pi2ur::piextQueueCreate)
  _PI_CL(piextQueueCreate2, pi2ur::piextQueueCreate2)
  _PI_CL(piQueueGetInfo, pi2ur::piQueueGetInfo)
  _PI_CL(piQueueFinish, pi2ur::piQueueFinish)
  _PI_CL(piQueueFlush, pi2ur::piQueueFlush)
  _PI_CL(piQueueRetain, pi2ur::piQueueRetain)
  _PI_CL(piQueueRelease, pi2ur::piQueueRelease)
  _PI_CL(piextQueueGetNativeHandle, pi2ur::piextQueueGetNativeHandle)
  _PI_CL(piextQueueGetNativeHandle2, pi2ur::piextQueueGetNativeHandle2)
  _PI_CL(piextQueueCreateWithNativeHandle,
         pi2ur::piextQueueCreateWithNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle2,
         pi2ur::piextQueueCreateWithNativeHandle2)
  // Memory
  _PI_CL(piMemBufferCreate, pi2ur::piMemBufferCreate)
  _PI_CL(piMemImageCreate, pi2ur::piMemImageCreate)
  _PI_CL(piMemGetInfo, pi2ur::piMemGetInfo)
  _PI_CL(piMemImageGetInfo, pi2ur::piMemImageGetInfo)
  _PI_CL(piMemRetain, pi2ur::piMemRetain)
  _PI_CL(piMemRelease, pi2ur::piMemRelease)
  _PI_CL(piMemBufferPartition, pi2ur::piMemBufferPartition)
  _PI_CL(piextMemGetNativeHandle, pi2ur::piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, pi2ur::piextMemCreateWithNativeHandle)
  // Program
  _PI_CL(piProgramCreate, pi2ur::piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, pi2ur::piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, pi2ur::piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, pi2ur::piProgramGetInfo)
  _PI_CL(piProgramCompile, pi2ur::piProgramCompile)
  _PI_CL(piProgramBuild, pi2ur::piProgramBuild)
  _PI_CL(piProgramLink, pi2ur::piProgramLink)
  _PI_CL(piProgramGetBuildInfo, pi2ur::piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, pi2ur::piProgramRetain)
  _PI_CL(piProgramRelease, pi2ur::piProgramRelease)
  _PI_CL(piextProgramGetNativeHandle, pi2ur::piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle,
         pi2ur::piextProgramCreateWithNativeHandle)
  _PI_CL(piextProgramSetSpecializationConstant,
         pi2ur::piextProgramSetSpecializationConstant)
  // Kernel
  _PI_CL(piKernelCreate, pi2ur::piKernelCreate)
  _PI_CL(piKernelSetArg, pi2ur::piKernelSetArg)
  _PI_CL(piKernelGetInfo, pi2ur::piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, pi2ur::piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, pi2ur::piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, pi2ur::piKernelRetain)
  _PI_CL(piKernelRelease, pi2ur::piKernelRelease)
  _PI_CL(piKernelSetExecInfo, pi2ur::piKernelSetExecInfo)
  _PI_CL(piextKernelSetArgPointer, pi2ur::piKernelSetArgPointer)
  // Event
  _PI_CL(piEventCreate, pi2ur::piEventCreate)
  _PI_CL(piEventGetInfo, pi2ur::piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, pi2ur::piEventGetProfilingInfo)
  _PI_CL(piEventsWait, pi2ur::piEventsWait)
  _PI_CL(piEventSetCallback, pi2ur::piEventSetCallback)
  _PI_CL(piEventSetStatus, pi2ur::piEventSetStatus)
  _PI_CL(piEventRetain, pi2ur::piEventRetain)
  _PI_CL(piEventRelease, pi2ur::piEventRelease)
  _PI_CL(piextEventGetNativeHandle, pi2ur::piextEventGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle,
         pi2ur::piextEventCreateWithNativeHandle)
  // Sampler
  _PI_CL(piSamplerCreate, pi2ur::piSamplerCreate)
  _PI_CL(piSamplerGetInfo, pi2ur::piSamplerGetInfo)
  _PI_CL(piSamplerRetain, pi2ur::piSamplerRetain)
  _PI_CL(piSamplerRelease, pi2ur::piSamplerRelease)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, hip_piEnqueueKernelLaunch)
  _PI_CL(piEnqueueNativeKernel, hip_piEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, hip_piEnqueueEventsWait)
  _PI_CL(piEnqueueEventsWaitWithBarrier, hip_piEnqueueEventsWaitWithBarrier)
  _PI_CL(piEnqueueMemBufferRead, hip_piEnqueueMemBufferRead)
  _PI_CL(piEnqueueMemBufferReadRect, hip_piEnqueueMemBufferReadRect)
  _PI_CL(piEnqueueMemBufferWrite, hip_piEnqueueMemBufferWrite)
  _PI_CL(piEnqueueMemBufferWriteRect, hip_piEnqueueMemBufferWriteRect)
  _PI_CL(piEnqueueMemBufferCopy, hip_piEnqueueMemBufferCopy)
  _PI_CL(piEnqueueMemBufferCopyRect, hip_piEnqueueMemBufferCopyRect)
  _PI_CL(piEnqueueMemBufferFill, hip_piEnqueueMemBufferFill)
  _PI_CL(piEnqueueMemImageRead, hip_piEnqueueMemImageRead)
  _PI_CL(piEnqueueMemImageWrite, hip_piEnqueueMemImageWrite)
  _PI_CL(piEnqueueMemImageCopy, hip_piEnqueueMemImageCopy)
  _PI_CL(piEnqueueMemImageFill, hip_piEnqueueMemImageFill)
  _PI_CL(piEnqueueMemBufferMap, hip_piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, hip_piEnqueueMemUnmap)
  // USM
  _PI_CL(piextUSMHostAlloc, pi2ur::piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, pi2ur::piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, pi2ur::piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, pi2ur::piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, hip_piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, hip_piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, hip_piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, hip_piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMEnqueueMemcpy2D, hip_piextUSMEnqueueMemcpy2D)
  _PI_CL(piextUSMEnqueueFill2D, hip_piextUSMEnqueueFill2D)
  _PI_CL(piextUSMEnqueueMemset2D, hip_piextUSMEnqueueMemset2D)
  _PI_CL(piextUSMGetMemAllocInfo, pi2ur::piextUSMGetMemAllocInfo)
  // Device global variable
  _PI_CL(piextEnqueueDeviceGlobalVariableWrite,
         hip_piextEnqueueDeviceGlobalVariableWrite)
  _PI_CL(piextEnqueueDeviceGlobalVariableRead,
         hip_piextEnqueueDeviceGlobalVariableRead)

  // Host Pipe
  _PI_CL(piextEnqueueReadHostPipe, hip_piextEnqueueReadHostPipe)
  _PI_CL(piextEnqueueWriteHostPipe, hip_piextEnqueueWriteHostPipe)

  _PI_CL(piextKernelSetArgMemObj, hip_piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, hip_piextKernelSetArgSampler)
  _PI_CL(piPluginGetLastError, hip_piPluginGetLastError)
  _PI_CL(piTearDown, pi2ur::piTearDown)
  _PI_CL(piGetDeviceAndHostTimer, hip_piGetDeviceAndHostTimer)
  _PI_CL(piPluginGetBackendOption, hip_piPluginGetBackendOption)

#undef _PI_CL

  return PI_SUCCESS;
}

#ifdef _WIN32
#define __SYCL_PLUGIN_DLL_NAME "pi_hip.dll"
#include "../common_win_pi_trace/common_win_pi_trace.hpp"
#undef __SYCL_PLUGIN_DLL_NAME
#endif

} // extern "C"

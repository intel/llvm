//==---------- pi_cuda.cpp - CUDA Plugin -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/backend/cuda.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <pi_cuda.hpp>
#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <memory>
#include <limits>
#include <mutex>

std::string getCudaVersionString() {
  int driver_version = 0;
  cuDriverGetVersion(&driver_version);
  // The version is returned as (1000 major + 10 minor).
  std::stringstream stream;
  stream << "CUDA " << driver_version / 1000 << "." << driver_version % 100;
  return stream.str();
}

pi_result map_error(CUresult result) {
  switch (result) {
  case CUDA_SUCCESS:
    return PI_SUCCESS;
  case CUDA_ERROR_NOT_PERMITTED:
    return PI_INVALID_OPERATION;
  case CUDA_ERROR_INVALID_CONTEXT:
    return PI_INVALID_CONTEXT;
  case CUDA_ERROR_INVALID_DEVICE:
    return PI_INVALID_DEVICE;
  case CUDA_ERROR_INVALID_VALUE:
    return PI_INVALID_VALUE;
  case CUDA_ERROR_OUT_OF_MEMORY:
    return PI_OUT_OF_HOST_MEMORY;
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return PI_OUT_OF_RESOURCES;
  default:
    return PI_ERROR_UNKNOWN;
  }
}

inline void assign_result(pi_result *ptr, pi_result value) noexcept {
  if (ptr) {
    *ptr = value;
  }
}

pi_result check_error(CUresult result, const char *function, int line,
                      const char *file) {
  if (result == CUDA_SUCCESS) {
    return PI_SUCCESS;
  }

  const char *errorString = nullptr;
  const char *errorName = nullptr;
  cuGetErrorName(result, &errorName);
  cuGetErrorString(result, &errorString);
  std::cerr << "\nPI CUDA ERROR:"
            << "\n\tValue:           " << result
            << "\n\tName:            " << errorName
            << "\n\tDescription:     " << errorString
            << "\n\tFunction:        " << function
            << "\n\tSource Location: " << file << ":" << line << "\n"
            << std::endl;

  if(std::getenv("PI_CUDA_ABORT") != nullptr)
  {
    std::abort();
  }

  throw map_error(result);
}

#define PI_CHECK_ERROR(result) \
check_error(result, __func__, __LINE__, __FILE__)

//--------------
// PI object implementation

extern "C" {

// Required in a number of functions, so forward declare here
pi_result cuda_piEnqueueEventsWait(pi_queue command_queue,
                                   pi_uint32 num_events_in_wait_list,
                                   const pi_event *event_wait_list,
                                   pi_event *event);
pi_result cuda_piEventRelease(pi_event event);
pi_result cuda_piEventRetain(pi_event event);

} // extern "C"

_pi_event::_pi_event(pi_command_type type, pi_context context, pi_queue queue)
    : commandType_{type}, refCount_{1}, isCompleted_{false},
      isRecorded_{false},
      isStarted_{false}, event_{nullptr}, queue_{queue}, context_{context} {

  if (is_native_event()) {
    PI_CHECK_ERROR(cuEventCreate(&event_, 0));
    PI_CHECK_ERROR(cuEventCreate(&evStart_, 0));
  }

  
  if (queue_ != nullptr) {
    cuda_piQueueRetain(queue_);
  }
  cuda_piContextRetain(context_);
}

_pi_event::~_pi_event() {
  if (queue_ != nullptr) {
    cuda_piQueueRelease(queue_);
  }
  cuda_piContextRelease(context_);
}



pi_result _pi_event::start() {
  assert(!is_started());
  pi_result result;

  try {
    if (is_native_event()) {
      result = PI_CHECK_ERROR(cuEventRecord(evStart_, queue_->get()));
    }
  } catch (pi_result error) {
    result = error;
  }

  isStarted_ = true;
  return result;
}

pi_uint64 _pi_event::get_end_time() const {
  float miliSeconds = 0.0f;
  assert(is_started() && is_recorded());

  PI_CHECK_ERROR(cuEventElapsedTime(&miliSeconds, evStart_, event_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_result _pi_event::record() {

  if (is_recorded()) {
    return PI_INVALID_EVENT;
  }

  pi_result result = PI_INVALID_OPERATION;

  if (is_native_event()) {

    if (!queue_) {
      return PI_INVALID_QUEUE;
    }

    CUstream cuStream = queue_->get();

    try {
      result = PI_CHECK_ERROR(cuEventRecord(event_, cuStream));
    } catch (pi_result error) {
      result = error;
    }
  } else {
    result = PI_SUCCESS;
  }

  if (result == PI_SUCCESS) {
    isRecorded_ = true;
  }

  return result;
}

pi_result _pi_event::wait() {

  pi_result retErr;
  if (is_native_event()) {
    try {
      retErr = PI_CHECK_ERROR(cuEventSynchronize(event_));
    } catch (pi_result error) {
      retErr = error;
    }
  } else {

    while (!is_completed()) {
      // wait for user event to complete
    }
    retErr = PI_SUCCESS;
  }

  return retErr;
}

pi_event_status _pi_event::get_execution_status() const noexcept {

  if (!is_recorded()) {
    return PI_EVENT_SUBMITTED;
  }

  if (is_native_event()) {
    // native event status

    auto status = cuEventQuery(get());
    if (status == CUDA_ERROR_NOT_READY) {
      return PI_EVENT_RUNNING;
    } else if (status != CUDA_SUCCESS) {
      cl::sycl::detail::pi::die("Invalid CUDA event status");
    }
    return PI_EVENT_COMPLETE;
  } else {
    // user event status

    return is_completed() ? PI_EVENT_COMPLETE : PI_EVENT_RUNNING;
  }
}

// iterates over the event wait list, returns correct pi_result error codes.
// Invokes the callback for each event in the wait list. The callback must take
// a single pi_event argument and return a pi_result.
template <typename Func>
pi_result forEachEvent(const pi_event *event_wait_list,
                       std::size_t num_events_in_wait_list, Func &&f) {

  if (event_wait_list == nullptr || num_events_in_wait_list == 0) {
    return PI_INVALID_EVENT_WAIT_LIST;
  }

  for (size_t i = 0; i < num_events_in_wait_list; i++) {
    auto event = event_wait_list[i];
    if (event == nullptr) {
      return PI_INVALID_EVENT_WAIT_LIST;
    }

    auto result = f(event);
    if (result != PI_SUCCESS) {
      return result;
    }
  }

  return PI_SUCCESS;
}

// makes all future work submitted to queue wait for all work captured in event.
pi_result enqueueEventWait(pi_queue queue, pi_event event) {
  if (event->is_native_event()) {

    // for native events, the cuStreamWaitEvent call is used.
    // This makes all future work submitted to stream wait for all
    // work captured in event.

    return PI_CHECK_ERROR(cuStreamWaitEvent(queue->get(), event->get(), 0));

  } else {

    // for user events, we enqueue a callback. When invoked, the
    // callback will block until the user event is marked as
    // completed.

    static auto user_wait_func = [](void *user_data) {
      // The host function must not make any CUDA API calls.
      auto event = static_cast<pi_event>(user_data);

      // busy wait for user event to complete
      event->wait();

      // this function does not need the event to be kept alive
      // anymore
      cuda_piEventRelease(event);
    };

    // retain event to ensure it is still alive when the
    // user_wait_func callback is invoked
    cuda_piEventRetain(event);

    return PI_CHECK_ERROR(cuLaunchHostFunc(queue->get(), user_wait_func, event));
  }
}

_pi_program::_pi_program(pi_context ctxt)
    : module_{nullptr}, source_{}, sourceLength_{0}
    , refCount_{1}, context_{ctxt} 
{
  cuda_piContextRetain(context_);
}

_pi_program::~_pi_program() {
  cuda_piContextRelease(context_);
}

pi_result _pi_program::create_from_source(const char *source, size_t length) {
  source_ = source;
  sourceLength_ = length;
  return PI_SUCCESS;
}

pi_result _pi_program::build_program(const char *build_options) {

  this->buildOptions_ = build_options;

  constexpr const unsigned int numberOfOptions = 4u;

  CUjit_option options[numberOfOptions];
  void *optionVals[numberOfOptions];

  // Pass a buffer for info messages
  options[0] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[0] = (void *)infoLog_;
  // Pass the size of the info buffer
  options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[1] = (void *)(long)MAX_LOG_SIZE;
  // Pass a buffer for error message
  options[2] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[2] = (void *)errorLog_;
  // Pass the size of the error buffer
  options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[3] = (void *)(long)MAX_LOG_SIZE;

  auto result = PI_CHECK_ERROR(cuModuleLoadDataEx(
      &module_, static_cast<const void *>(source_), numberOfOptions, options,
      optionVals));

  const auto success = (result == PI_SUCCESS);

  buildStatus_ =
      success ? PI_PROGRAM_BUILD_STATUS_SUCCESS : PI_PROGRAM_BUILD_STATUS_ERROR;

  // If no exception, result is correct
  return success ? PI_SUCCESS : PI_BUILD_PROGRAM_FAILURE;
}

namespace cl {
namespace sycl {
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

void assertion(bool Condition, const char *Message) {
  if (!Condition)
    die(Message);
}

}  // namespace pi
}  // namespace detail
}  // namespace sycl
}  // namespace cl

// RAII type to guarantee recovering original CUDA context
class ScopedContext {
  pi_context placedContext_;
  CUcontext original_;
  bool needToRecover_;

public:
  ScopedContext(pi_context ctxt) : placedContext_{ctxt}, needToRecover_{false} {

    if (!placedContext_) {
      throw PI_INVALID_CONTEXT;
    }

    CUcontext desired = placedContext_->get();
    PI_CHECK_ERROR(cuCtxGetCurrent(&original_));
    if (original_ != desired) {
      // Sets the desired context as the active one for the thread
      PI_CHECK_ERROR(cuCtxSetCurrent(desired));
      if (original_ == nullptr && ctxt->is_primary()) {
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying CUDA primary context are destroyed. This emulates
        // the behaviour of the CUDA runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
      } else {
        needToRecover_ = true;
      }
    }
  }

  ~ScopedContext() {
    if (needToRecover_) {
      PI_CHECK_ERROR(cuCtxSetCurrent(original_));
    }
  }
};

template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return PI_INVALID_VALUE;
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
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

/// RAII object that calls the reference count release function on the held PI
/// object on destruction.
///
/// The `dismiss` function stops the release from happening on destruction.
template <typename T> class ReleaseGuard {
private:
  T Captive;

  static pi_result callRelease(pi_device Captive) {
    return cuda_piDeviceRelease(Captive);
  }

  static pi_result callRelease(pi_context Captive) {
    return cuda_piContextRelease(Captive);
  }

  static pi_result callRelease(pi_mem Captive) {
    return cuda_piMemRelease(Captive);
  }

  static pi_result callRelease(pi_program Captive) {
    return cuda_piProgramRelease(Captive);
  }

  static pi_result callRelease(pi_kernel Captive) {
    return cuda_piKernelRelease(Captive);
  }

  static pi_result callRelease(pi_queue Captive) {
    return cuda_piQueueRelease(Captive);
  }

  static pi_result callRelease(pi_event Captive) {
    return cuda_piEventRelease(Captive);
  }

public:
  ReleaseGuard() = delete;
  /// Obj can be `nullptr`.
  explicit ReleaseGuard(T Obj) : Captive(Obj) {}
  ReleaseGuard(ReleaseGuard &&Other) noexcept : Captive(Other.Captive) {
    Other.Captive = nullptr;
  }

  ReleaseGuard(const ReleaseGuard &) = delete;

  /// Calls the related PI object release function if the object held is not
  /// `nullptr` or if `dismiss` has not been called.
  ~ReleaseGuard() {
    if (Captive != nullptr) {
      pi_result ret = callRelease(Captive);
      if (ret != PI_SUCCESS) {
        // A reported CUDA error is either an implementation or an asynchronous
        // CUDA error for which it is unclear if the function that reported it
        // succeeded or not. Either way, the state of the program is compromised
        // and likely unrecoverable.
        cl::sycl::detail::pi::die("Unrecoverable program state reached in cuda_piMemRelease");
      }
    }
  }

  ReleaseGuard &operator=(const ReleaseGuard &) = delete;

  ReleaseGuard &operator=(ReleaseGuard &&Other) {
    Captive = Other.Captive;
    Other.Captive = nullptr;
    return *this;
  }

  /// End the guard and do not release the reference count of the held
  /// PI object.
  void dismiss() { Captive = nullptr; }
};

//-- PI API implementation
extern "C" {

pi_result cuda_piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                              pi_uint32 *num_platforms) {

  try {
    static constexpr pi_uint32 numPlatforms = 1;

    if (num_platforms != nullptr) {
      *num_platforms = numPlatforms;
    }

    pi_result err = PI_SUCCESS;

    if (platforms != nullptr) {

      assert(num_entries != 0);

      static std::once_flag initFlag;
      static _pi_platform platformId;
      std::call_once(initFlag,
                     [](pi_result &err) { err = PI_CHECK_ERROR(cuInit(0)); },
                     err);

      *platforms = &platformId;
    }

    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result cuda_piPlatformGetInfo(pi_platform platform,
                                 pi_platform_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  assert(platform != nullptr);

  switch (param_name) {
  case PI_PLATFORM_INFO_NAME:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "NVIDIA CUDA");
  case PI_PLATFORM_INFO_VENDOR:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "NVIDIA Corporation");
  case PI_PLATFORM_INFO_PROFILE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "FULL PROFILE");
  case PI_PLATFORM_INFO_VERSION: {
    auto version = getCudaVersionString();
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   version.c_str());
  }
  case PI_PLATFORM_INFO_EXTENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Platform info request not implemented");
  return {};
}

pi_result cuda_piDevicesGet(pi_platform platform, pi_device_type device_type,
                            pi_uint32 num_entries, pi_device *devices,
                            pi_uint32 *num_devices) {

  pi_result err = PI_SUCCESS;
  const bool askingForGPU = (device_type & PI_DEVICE_TYPE_GPU);
  size_t numDevices = askingForGPU ? 1 : 0;

  try {
    if (num_devices) {
      *num_devices = numDevices;
    }

    if (askingForGPU) {
      if (devices) {
        CUdevice device;
        err = PI_CHECK_ERROR(cuDeviceGet(&device, 0));
        *devices = new _pi_device{device, platform};
      }
    } else {
      if (devices) {
        *devices = nullptr;
      }
    }

    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result cuda_piDeviceRetain(pi_device device) {
  // OpenCL: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clRetainDevice.html
  // Returns CL_SUCCESS if the function is executed successfully or the device is a root-level device.
  return PI_SUCCESS;
}

pi_result cuda_piContextGetInfo(pi_context context, pi_context_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {

  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  case PI_CONTEXT_INFO_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   context->get_device());
  case PI_CONTEXT_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   context->get_reference_count());
  }

  return PI_OUT_OF_RESOURCES;
}

pi_result cuda_piContextRetain(pi_context context) {
  assert(context != nullptr);
  assert(context->get_reference_count() > 0);

  context->increment_reference_count();
  return PI_SUCCESS;
}

pi_result cuda_piDevicePartition(
    pi_device device,
    const cl_device_partition_property *properties, // TODO: untie from OpenCL
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices) {
  return {};
}

pi_result cuda_piextDeviceSelectBinary(
    pi_device device, // TODO: does this need to be context?
    pi_device_binary *binaries, pi_uint32 num_binaries,
    pi_device_binary *selected_binary) {
  if (!binaries) {
    cl::sycl::detail::pi::die("No list of device images provided");
  }
  if (num_binaries < 1) {
    cl::sycl::detail::pi::die("No binary images in the list");
  }
  if (!selected_binary) {
    cl::sycl::detail::pi::die("No storage for device binary provided");
  }
  *selected_binary = binaries[0];
  return PI_SUCCESS;
}

pi_result cuda_piextGetDeviceFunctionPointer(pi_device device,
                                       pi_device_binary *binaries,
                                       pi_uint32 num_binaries,
                                       pi_device_binary *selected_binary) {
  cl::sycl::detail::pi::die("cuda_piextGetDeviceFunctionPointer not implemented");
  return {};
}

pi_result cuda_piDeviceRelease(pi_device device) {
  // OpenCL: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clReleaseDevice.html
  // If device is a root level device i.e. a cl_device_id returned by clGetDeviceIDs, the device reference count remains unchanged.
  return PI_SUCCESS;
}

pi_result cuda_piDeviceGetInfo(pi_device device, pi_device_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {

  static constexpr pi_uint32 max_work_item_dimensions = 3u;

  assert(device != nullptr);

  switch (param_name) {
  case PI_DEVICE_INFO_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_DEVICE_TYPE_GPU);
  }
  case PI_DEVICE_INFO_VENDOR_ID: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 4318u);
  }
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    int compute_units = 0;
    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&compute_units,
                                       CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                       device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(compute_units >= 0);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint32(compute_units));
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   max_work_item_dimensions);
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    size_t return_sizes[max_work_item_dimensions];

    int max_x = 0, max_y = 0, max_z = 0;
    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&max_x,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                       device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(max_x >= 0);

    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&max_y,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                       device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(max_y >= 0);

    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&max_z,
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                                       device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(max_z >= 0);

    return_sizes[0] = size_t(max_x);
    return_sizes[1] = size_t(max_y);
    return_sizes[2] = size_t(max_z);
    return getInfoArray(max_work_item_dimensions, param_value_size, param_value,
                        param_value_size_ret, return_sizes);
  }
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    int max_work_group_size = 0;
    cl::sycl::detail::pi::assertion(
        cuDeviceGetAttribute(&max_work_group_size,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             device->get()) == CUDA_SUCCESS);

    cl::sycl::detail::pi::assertion(max_work_group_size >= 0);

    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(max_work_group_size));
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    int clock_freq = 0;
    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&clock_freq,
                                       CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                       device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(clock_freq >= 0);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint32(clock_freq) / 1000u);
  }
  case PI_DEVICE_INFO_ADDRESS_BITS: {
    auto bits = pi_uint32{std::numeric_limits<uintptr_t>::digits};
    return getInfo(param_value_size, param_value, param_value_size_ret, bits);
  }
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    // Max size of memory object allocation in bytes.
    // The minimum value is max(min(1024 × 1024 ×
    // 1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE),
    // 32 × 1024 × 1024) for devices that are not of type
    // CL_DEVICE_TYPE_CUSTOM.

    size_t global = 0;
    cl::sycl::detail::pi::assertion(cuDeviceTotalMem(&global, device->get()) == CUDA_SUCCESS);

    auto quarter_global = static_cast<pi_uint32>(global / 4u);

    auto max_alloc = std::max(std::min(1024u * 1024u * 1024u, quarter_global),
                              32u * 1024u * 1024u);

    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{max_alloc});
  }
  case PI_DEVICE_INFO_IMAGE_SUPPORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, false);
  }
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0);
  }
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_MAX_SAMPLERS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters
    // __global__ function parameters are passed to the device via constant
    // memory and are limited to 4 KB.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{4000u});
  }
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    // "The minimum value is the size (in bits) of the largest OpenCL built-in
    // data type supported by the device"
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
    // 1060 3GB
    return getInfo(param_value_size, param_value, param_value_size_ret, 4096u);
  }
  case PI_DEVICE_INFO_HALF_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    auto config = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                  CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA |
                  CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    auto config = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                  CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    // TODO: is this config consistent across all NVIDIA GPUs?
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   CL_READ_WRITE_CACHE);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // The value is documented for all existing GPUs in the CUDA programming
    // guidelines, section "H.3.2. Global Memory".
    return getInfo(param_value_size, param_value, param_value_size_ret, 128u);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    int cache_size = 0;
    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&cache_size,
                                       CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                                       device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(cache_size >= 0);
    // The L2 cache is global to the GPU.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(cache_size));
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    size_t bytes = 0;
    // Runtime API has easy access to this value, driver API info is scarse.
    cl::sycl::detail::pi::assertion(cuDeviceTotalMem(&bytes, device->get()) == CUDA_SUCCESS);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{bytes});
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    int constant_memory = 0;
    cl::sycl::detail::pi::assertion(
        cuDeviceGetAttribute(&constant_memory,
                             CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                             device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(constant_memory >= 0);

    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(constant_memory));
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: is there a way to retrieve this from CUDA driver API?
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
    // 1060 3GB
    return getInfo(param_value_size, param_value, param_value_size_ret, 9u);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_LOCAL_MEM_TYPE_LOCAL);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE: {
    // OpenCL's "local memory" maps most closely to CUDA's "shared memory".
    // CUDA has its own definition of "local memory", which maps to OpenCL's
    // "private memory".
    int local_mem_size = 0;
    cl::sycl::detail::pi::assertion(
        cuDeviceGetAttribute(&local_mem_size,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                             device->get()) == CUDA_SUCCESS);
    cl::sycl::detail::pi::assertion(local_mem_size >= 0);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(local_mem_size));
  }
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    int ecc_enabled = 0;
    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&ecc_enabled,
                                       CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
                                       device->get()) == CUDA_SUCCESS);

    cl::sycl::detail::pi::assertion((ecc_enabled == 0) | (ecc_enabled == 1));
    auto result = static_cast<bool>(ecc_enabled);
    return getInfo(param_value_size, param_value, param_value_size_ret, result);
  }
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    int is_integrated = 0;
    cl::sycl::detail::pi::assertion(cuDeviceGetAttribute(&is_integrated,
                                       CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                       device->get()) == CUDA_SUCCESS);

    cl::sycl::detail::pi::assertion((is_integrated == 0) | (is_integrated == 1));
    auto result = static_cast<bool>(is_integrated);
    return getInfo(param_value_size, param_value, param_value_size_ret, result);
  }
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Hard coded to value returned by clinfo for OpenCL 1.2 CUDA | GeForce GTX
    // 1060 3GB
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1000u});
  }
  case PI_DEVICE_INFO_IS_ENDIAN_LITTLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_IS_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_IS_COMPILER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_IS_LINKER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    auto capability = CL_EXEC_KERNEL;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    auto capability = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
    // The mandated minimum capability:
    auto capability = CL_QUEUE_PROFILING_ENABLE;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_PLATFORM: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   device->platform_);
  }
  case PI_DEVICE_INFO_NAME: {
    static constexpr size_t MAX_DEVICE_NAME_LENGTH = 256u;
    char name[MAX_DEVICE_NAME_LENGTH];
    cl::sycl::detail::pi::assertion(cuDeviceGetName(name, MAX_DEVICE_NAME_LENGTH,
                                  device->get()) == CUDA_SUCCESS);
    return getInfoArray(strlen(name) + 1, param_value_size, param_value,
                        param_value_size_ret, name);
  }
  case PI_DEVICE_INFO_VENDOR: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "NVIDIA Corporation");
  }
  case PI_DEVICE_INFO_DRIVER_VERSION: {
    auto version = getCudaVersionString();
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   version.c_str());
  }
  case PI_DEVICE_INFO_PROFILE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "CUDA");
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   device->get_reference_count());
  }
  case PI_DEVICE_INFO_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "PI 0.0");
  }
  case PI_DEVICE_INFO_OPENCL_C_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_EXTENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    // The minimum value for the FULL profile is 1 MB.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1024u});
  }
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_PARENT_DEVICE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  }
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<cl_device_partition_property>(0u));
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_PARTITION_TYPE: {
    // TODO: uncouple from OpenCL
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<cl_device_partition_property>(0u));
  }
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Device info request not implemented");
  return {};
}

/* Context APIs */
pi_result cuda_piContextCreate(const cl_context_properties *properties,
                                pi_uint32 num_devices, const pi_device *devices,
                                void (*pfn_notify)(const char *errinfo,
                                                   const void *private_info,
                                                   size_t cb, void *user_data),
                                void *user_data, pi_context *retcontext) {

  assert(devices != nullptr);
  // TODO: How to implement context callback?
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  // assert(properties == nullptr);
  assert(num_devices == 1);
  // Need input context
  assert(retcontext != nullptr);
  pi_result errcode_ret = PI_SUCCESS;

  std::unique_ptr<_pi_context> piContextPtr{nullptr};
  try {
    if (properties && *properties != PI_CONTEXT_PROPERTIES_CUDA_PRIMARY) {
      throw pi_result(CL_INVALID_VALUE);
    } else if (!properties) {
      CUcontext newContext, current;
      PI_CHECK_ERROR(cuCtxGetCurrent(&current));
      errcode_ret = PI_CHECK_ERROR(cuCtxCreate(&newContext, CU_CTX_MAP_HOST,
                                            (*devices)->cuDevice_));
      piContextPtr = std::unique_ptr<_pi_context>(new _pi_context{
          _pi_context::kind::user_defined, newContext, *devices});
      if (current != nullptr) {
        // If there was an existing context on the thread we recover it
        PI_CHECK_ERROR(cuCtxSetCurrent(current));
      }
    } else if (properties 
                  && *properties == PI_CONTEXT_PROPERTIES_CUDA_PRIMARY) {
      CUcontext Ctxt;
      errcode_ret = PI_CHECK_ERROR(cuDevicePrimaryCtxRetain(
                                     &Ctxt, (*devices)->cuDevice_));
      piContextPtr = std::unique_ptr<_pi_context>(
          new _pi_context{_pi_context::kind::primary, Ctxt, *devices});
      errcode_ret = PI_CHECK_ERROR(cuCtxPushCurrent(Ctxt));
    } else {
      throw pi_result(CL_INVALID_VALUE);
    }

    *retcontext = piContextPtr.release();
  } catch (pi_result err) {
    errcode_ret = err;
  } catch (...) {
    errcode_ret = PI_OUT_OF_RESOURCES;
  }
  return errcode_ret;
}

pi_result cuda_piContextRelease(pi_context ctxt) {
  
  assert(ctxt != nullptr);

  if (ctxt->decrement_reference_count() > 0) {
    return PI_SUCCESS;
  }
  ctxt->invoke_callback();

  std::unique_ptr<_pi_context> context{ctxt};

  if (!ctxt->is_primary()) {
    CUcontext cuCtxt = ctxt->get();
    CUcontext current = nullptr;
    cuCtxGetCurrent(&current);
    if(cuCtxt != current)
    {
      PI_CHECK_ERROR(cuCtxSetCurrent(cuCtxt));
    }
    PI_CHECK_ERROR(cuCtxSynchronize());
    return PI_CHECK_ERROR(cuCtxDestroy(cuCtxt));
  } else {
    // Primary context is not destroyed, but released
    CUdevice cuDev = ctxt->get_device()->get();
    CUcontext current;
    cuCtxPopCurrent(&current);
    return PI_CHECK_ERROR(cuDevicePrimaryCtxRelease(cuDev));
  }
}

pi_result cuda_piMemBufferCreate(pi_context context, pi_mem_flags flags,
                              size_t size, void *host_ptr,
                              pi_mem *ret_mem) {
  // Need input memory object
  assert(ret_mem != nullptr);
  // Currently, USE_HOST_PTR is not implemented using host register
  // since this triggers a weird segfault after program ends.
  // Setting this constant to true enables testing that behavior.
  const bool enableUseHostPtr = false;
  const bool performInitialCopy = (flags & PI_MEM_FLAGS_HOST_PTR_COPY) 
    || ((flags & PI_MEM_FLAGS_HOST_PTR_USE) && !enableUseHostPtr);
  pi_result retErr = PI_SUCCESS;
  pi_mem retMemObj = nullptr;

  try {
    ScopedContext active(context);
    CUdeviceptr ptr;
    _pi_mem::alloc_mode allocMode = _pi_mem::alloc_mode::classic;


    if ((flags & PI_MEM_FLAGS_HOST_PTR_USE) && enableUseHostPtr) {
      retErr = PI_CHECK_ERROR(cuMemHostRegister(host_ptr, size, 
                            CU_MEMHOSTREGISTER_DEVICEMAP));
      retErr = PI_CHECK_ERROR(cuMemHostGetDevicePointer(&ptr, host_ptr, 0));
      allocMode  = _pi_mem::alloc_mode::use_host_ptr;
    } else {
      retErr = PI_CHECK_ERROR(cuMemAlloc(&ptr, size));
    }

    if (retErr == PI_SUCCESS) {
      pi_mem parentBuffer = nullptr;

      auto piMemObj = std::unique_ptr<_pi_mem>(
          new _pi_mem{context, parentBuffer, allocMode, ptr, host_ptr, size});
      if (piMemObj != nullptr) {
        retMemObj = piMemObj.release();
        if (performInitialCopy) {
          retErr = PI_CHECK_ERROR(cuMemcpyHtoD(ptr, host_ptr, size));
        }
      } else {
        retErr = PI_OUT_OF_HOST_MEMORY;
      }
    } 
  } catch (pi_result err) {
    retErr = err;
  } catch (...) {
    retErr = PI_OUT_OF_RESOURCES;
  }

  *ret_mem = retMemObj;

  return retErr;
}

pi_result cuda_piMemRelease(pi_mem memObj) {
  assert((memObj != nullptr) && "PI_INVALID_MEM_OBJECTS");

  pi_result ret = PI_SUCCESS;

  try {
    // Do nothing if there are other references
    if (memObj->decrement_reference_count() > 0) {
      return PI_SUCCESS;
    }

    // make sure memObj is released in case PI_CHECK_ERROR throws
    std::unique_ptr<_pi_mem> uniqueMemObj(memObj);

    if (!memObj->is_sub_buffer()) {

      ScopedContext(uniqueMemObj->get_context());

      switch (uniqueMemObj->allocMode_) {
        case _pi_mem::alloc_mode::classic:
          ret = PI_CHECK_ERROR(cuMemFree(uniqueMemObj->ptr_));
          break;
        case _pi_mem::alloc_mode::use_host_ptr:
          ret = PI_CHECK_ERROR(cuMemHostUnregister(uniqueMemObj->hostPtr_));
          break;
      };
    }

  } catch (pi_result err) {
    ret = err;
  } catch (...) {
    ret = PI_OUT_OF_RESOURCES;
  }

  if (ret != PI_SUCCESS) {
    // A reported CUDA error is either an implementation or an asynchronous CUDA
    // error for which it is unclear if the function that reported it succeeded
    // or not. Either way, the state of the program is compromised and likely
    // unrecoverable.
    cl::sycl::detail::pi::die("Unrecoverable program state reached in cuda_piMemRelease");
  }

  return PI_SUCCESS;
}

pi_result cuda_piMemBufferPartition(pi_mem parent_buffer, pi_mem_flags flags,
                                    pi_buffer_create_type buffer_create_type,
                                    void *buffer_create_info,
                                    pi_mem* memObj) {
  assert((parent_buffer != nullptr) && "PI_INVALID_MEM_OBJECT");
  assert(parent_buffer->is_buffer() && "PI_INVALID_MEM_OBJECTS");
  assert(!parent_buffer->is_sub_buffer() && "PI_INVALID_MEM_OBJECT");

  // Default value for flags means PI_MEM_FLAGS_ACCCESS_RW.
  if (flags == 0) {
    flags = PI_MEM_FLAGS_ACCESS_RW;
  }

  assert((flags == PI_MEM_FLAGS_ACCESS_RW) && "PI_INVALID_VALUE");
  assert((buffer_create_type == PI_BUFFER_CREATE_TYPE_REGION) &&
         "PI_INVALID_VALUE");
  assert((buffer_create_info != nullptr) && "PI_INVALID_VALUE");
  assert(memObj != nullptr);

  const auto bufferRegion =
      *reinterpret_cast<const cl_buffer_region *>(buffer_create_info);
  assert((bufferRegion.size != 0u) && "PI_INVALID_BUFFER_SIZE");

  assert((bufferRegion.origin <= (bufferRegion.origin + bufferRegion.size)) &&
         "Overflow");
  assert(
    ((bufferRegion.origin + bufferRegion.size) <= parent_buffer->get_size()) &&
    "PI_INVALID_BUFFER_SIZE");
  // Retained indirectly due to retaining parent buffer below.
  pi_context context = parent_buffer->context_;
  _pi_mem::alloc_mode allocMode = _pi_mem::alloc_mode::classic;

  assert(parent_buffer->ptr_ != _pi_mem::native_type{0});
  _pi_mem::native_type ptr = parent_buffer->ptr_ + bufferRegion.origin;

  void *hostPtr = nullptr;
  if (parent_buffer->hostPtr_) {
    hostPtr =
      static_cast<char *>(parent_buffer->hostPtr_) + bufferRegion.origin;
  }

  // TODO: Enable once cuda_piDeviceGetInfo fix MR is merged.
  //
  // {
  //   // TODO: Add multi-device support if required.
  //   pi_device device = context->get_device();
  //   assert(device != nullptr);
  //   pi_uint32 requiredMinAlignment = 0;
  //   pi_result ret = cuda_piDeviceGetInfo(device, PI_DEVICE_MEM_BASE_ADDR_ALIGN,
  //                                        sizeof(requiredMinAlignment),
  //                                        &requiredMinAlignment, nullptr);
  //   assert(ret == PI_SUCCESS);
  //   (void)ret; // Suppress unused warning.
  //
  //   // TODO: Extract `is_aligned` helper function into common header.
  //   auto is_aligned = [](size_t value, size_t alignment) -> bool {
  //     assert((((alignment - 1u) & alignment) == 0u) &&
  //            "alignment must be a power of 2");
  //     return (value & (alignment - 1u)) == 0u;
  //   }
  //   (void)is_aligned; // Suppress unused warning.
  //
  //   auto OriginPtr = static_cast<size_t>(ptr);
  //   assert(is_aligned(OriginPtr, requiredMinAlignment) &&
  //          "PI_MISALIGNED_SUB_BUFFER_OFFSET");
  //   (void)OriginPtr; // Suppress unused warning.
  // }

  ReleaseGuard<pi_mem> releaseGuard(parent_buffer);

  std::unique_ptr<_pi_mem> retMemObj{nullptr};
  try {
    ScopedContext active(context);

    retMemObj = std::unique_ptr<_pi_mem>{
        new _pi_mem{context, parent_buffer, allocMode, ptr, hostPtr, 
                    bufferRegion.size}};
  } catch (pi_result err) {
    *memObj = nullptr;
    return err;
  } catch (...) {
    *memObj = nullptr;
    return PI_OUT_OF_HOST_MEMORY;
  }

  releaseGuard.dismiss();
  *memObj = retMemObj.release();
  return PI_SUCCESS;
}

pi_result cuda_piMemGetInfo(pi_mem memObj, cl_mem_info queriedInfo,
                            size_t expectedQuerySize, void *queryOutput,
                            size_t *writtenQuerySize) {

  cl::sycl::detail::pi::die("cuda_piMemGetInfo not implemented");
}

pi_result cuda_piQueueCreate(pi_context context, pi_device device,
                             pi_queue_properties properties, pi_queue *queue) {
  try {
    pi_result err = PI_SUCCESS;

    std::unique_ptr<_pi_queue> queueImpl{nullptr};

    if (context->get_device() != device) {
      *queue = nullptr;
      return PI_INVALID_DEVICE;
    }

    ScopedContext active(context);

    CUstream cuStream;
    unsigned int flags = 0;

    if (properties == PI_CUDA_USE_DEFAULT_STREAM) {
      flags = CU_STREAM_DEFAULT;
    } else if (properties == PI_CUDA_SYNC_WITH_DEFAULT) {
      flags = 0;
    } else {
      flags = CU_STREAM_NON_BLOCKING;
    }

    err = PI_CHECK_ERROR(cuStreamCreate(&cuStream, flags));
    if (err != PI_SUCCESS) {
      return err;
    }

    queueImpl = std::unique_ptr<_pi_queue>(
      new _pi_queue{cuStream, context, device, properties});
    
    *queue = queueImpl.release();

    return PI_SUCCESS;
  } catch (pi_result err) {

    return err;

  } catch (...) {

    return PI_OUT_OF_RESOURCES;
  }
}

pi_result cuda_piQueueGetInfo(pi_queue command_queue, pi_queue_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert(command_queue != nullptr);

  switch (param_name) {
  case PI_QUEUE_INFO_CONTEXT:
    return getInfo(param_value_size, param_value,
                               param_value_size_ret, command_queue->context_);
  case PI_QUEUE_INFO_DEVICE:
    return getInfo(param_value_size, param_value,
                              param_value_size_ret, command_queue->device_);
  case PI_QUEUE_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value,
                              param_value_size_ret,
                              command_queue->get_reference_count());
  case PI_QUEUE_INFO_PROPERTIES:
    return getInfo(param_value_size, param_value,
                                        param_value_size_ret,
                                        command_queue->properties_);
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Queue info request not implemented");
  return {};
}

pi_result cuda_piQueueRetain(pi_queue command_queue) {
  assert(command_queue != nullptr);
  assert(command_queue->get_reference_count() > 0);

  command_queue->increment_reference_count();
  return PI_SUCCESS;
}

pi_result cuda_piQueueRelease(pi_queue command_queue) {
  assert(command_queue != nullptr);

  if (command_queue->decrement_reference_count() > 0) {
    return PI_SUCCESS;
  }

  try {
    std::unique_ptr<_pi_queue> queueImpl(command_queue);
    
    ScopedContext active(command_queue->get_context());

    auto stream = queueImpl->stream_;
    PI_CHECK_ERROR(cuStreamSynchronize(stream));
    PI_CHECK_ERROR(cuStreamDestroy(stream));

    return PI_SUCCESS;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result cuda_piQueueFinish(pi_queue command_queue) {

  // set default result to a negative result (avoid false-positve tests)
  pi_result result = PI_OUT_OF_HOST_MEMORY;

  try {

    assert(command_queue !=
           nullptr); // need PI_ERROR_INVALID_EXTERNAL_HANDLE error code
    ScopedContext active(command_queue->get_context());
    result = PI_CHECK_ERROR(cuStreamSynchronize(command_queue->stream_));

  } catch (pi_result err) {

    result = err;

  } catch (...) {

    result = PI_OUT_OF_RESOURCES;
  }

  return result;
}

pi_result cuda_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                                       pi_bool blocking_write, size_t offset,
                                       size_t size, const void *ptr,
                                       pi_uint32 num_events_in_wait_list,
                                       const pi_event *event_wait_list,
                                       pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);
  pi_result retErr = PI_SUCCESS;
  CUstream cuStream = command_queue->get();
  CUdeviceptr devPtr = buffer->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
        event_wait_list, nullptr);
      
    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_WRITE, command_queue));
      retImplEv->start();
    }

    retErr = PI_CHECK_ERROR(cuMemcpyHtoDAsync(devPtr + offset, ptr, size, cuStream));

    if (event) {
      retErr = retImplEv->record();
    }

    if (blocking_write) {
      retErr = PI_CHECK_ERROR(cuStreamSynchronize(cuStream));
    }

    if (event) {
      *event = retImplEv.release();
    }
  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piEnqueueMemBufferRead(pi_queue command_queue, pi_mem buffer,
                                      pi_bool blocking_read, size_t offset,
                                      size_t size, void *ptr,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *retEvent) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);
  pi_result retErr = PI_SUCCESS;
  CUstream cuStream = command_queue->get();
  CUdeviceptr devPtr = buffer->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
        event_wait_list, nullptr);

    if (retEvent) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_READ, command_queue));
      retImplEv->start();
    }

    retErr = PI_CHECK_ERROR(cuMemcpyDtoHAsync(ptr, devPtr + offset, size, cuStream));

    if (retEvent) {
      retErr = retImplEv->record();
    }

    if (blocking_read) {
      retErr = PI_CHECK_ERROR(cuStreamSynchronize(cuStream));
    }

    if (retEvent) {
      *retEvent = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piEventsWait(pi_uint32 num_events, const pi_event *event_list) {

  try {
    pi_result err = PI_SUCCESS;

    if (num_events == 0) {
      return PI_INVALID_VALUE;
    }

    if (!event_list) {
      return PI_INVALID_EVENT;
    }

    auto context = event_list[0]->get_context();
    ScopedContext active(context);

    for (pi_uint32 count = 0; count < num_events && (err == PI_SUCCESS);
         count++) {

      auto event = event_list[count];

      if (!event) {
        return PI_INVALID_EVENT;
      }

      if (event->get_context() != context) {
        return PI_INVALID_CONTEXT;
      }

      err = event->wait();
    }
    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result cuda_piclProgramCreateWithSource(pi_context context, pi_uint32 count,
                                            const char **strings,
                                            const size_t *lengths,
                                            pi_program *program) {

  assert(context != nullptr);
  assert(strings != nullptr);
  assert(program != nullptr);

  pi_result retErr = PI_SUCCESS;

  if (count == 0) {
    retErr = PI_INVALID_PROGRAM;
    return retErr;
  }
  // TODO: Implement multiple sources
  assert(count == 1);

  std::unique_ptr<_pi_program> retProgram{new _pi_program{context}};

  auto has_length = (lengths != nullptr);
  size_t length = has_length ? lengths[0] : strlen(strings[0]) + 1;

  retProgram->create_from_source(strings[0], length);

  *program = retProgram.release();

  return retErr;
}

pi_result cuda_piProgramBuild(pi_program program, pi_uint32 num_devices,
                              const pi_device *device_list, const char *options,
                              void (*pfn_notify)(pi_program program,
                                                 void *user_data),
                              void *user_data) {

  assert(program != nullptr);
  assert(num_devices == 1 || num_devices == 0);
  assert(device_list != nullptr || num_devices == 0);
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  pi_result retError = PI_SUCCESS;

  try {
    ScopedContext active(program->get_context());

    program->build_program(options);

  } catch (pi_result err) {
    retError = err;
  }
  return retError;
}

pi_result cuda_piKernelCreate(pi_program program, const char *kernel_name,
                              pi_kernel *kernel) {
  assert(kernel != nullptr);
  assert(program != nullptr);

  pi_result retErr = PI_SUCCESS;
  std::unique_ptr<_pi_kernel> retKernel{nullptr};

  try {
    ScopedContext active(program->get_context());
    CUfunction cuFunc;
    retErr = PI_CHECK_ERROR(cuModuleGetFunction(
                                   &cuFunc, program->get(), kernel_name));

    retKernel = std::unique_ptr<_pi_kernel>(
        new _pi_kernel{cuFunc, kernel_name, program, program->get_context()});

  } catch (pi_result err) {
    retErr = err;
  } catch (...) {
    retErr = PI_OUT_OF_HOST_MEMORY;
  }

  *kernel = retKernel.release();
  return retErr;
}

pi_result cuda_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                              size_t arg_size, const void *arg_value) {

  assert(kernel != nullptr);
  pi_result retErr = PI_SUCCESS;
  try {
    if (arg_value) {
      kernel->set_kernel_arg(arg_index, arg_size, arg_value);
    } else {
      kernel->set_kernel_local_arg(arg_index, arg_size);
    }
  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                                       const pi_mem *arg_value) {

  assert(kernel != nullptr);
  assert(arg_value != nullptr);

  pi_result retErr = PI_SUCCESS;
  try {
    CUdeviceptr cuPtr = (*arg_value)->get();
    kernel->set_kernel_arg(arg_index, sizeof(CUdeviceptr), (void *)&cuPtr);
  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piEnqueueKernelLaunch(
    pi_queue command_queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  // Preconditions
  assert(command_queue != nullptr);
  assert(command_queue->get_context() == kernel->get_context());
  assert(kernel != nullptr);
  assert(work_dim > 0);
  assert(work_dim < 4);

  pi_result retError = PI_SUCCESS;
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    CUfunction cuFunc = kernel->get();
    CUstream cuStream = command_queue->get();

    retError = cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
        event_wait_list, nullptr);

    // Set the number of threads per block to the number of threads per warp
    // by default unless user has provided a better number
    int threadsPerBlock[3] = {32, 1, 1};

    if (local_work_size) {
      for (size_t i = 0; i < work_dim; i++) {
        threadsPerBlock[i] = static_cast<int>(local_work_size[i]);
      }
    } else {
       for (size_t i = 0; i < work_dim; i++) {
        threadsPerBlock[i] = std::min(static_cast<int>(global_work_size[i]), 
                                      static_cast<int>(threadsPerBlock[i]));
      }
    }

    int blocksPerGrid[3] = { 1, 1, 1 };

    for (size_t i = 0; i < work_dim; i++) {
      blocksPerGrid[i] = static_cast<int>(global_work_size[i]
                                + threadsPerBlock[i] - 1) / threadsPerBlock[i];
    }

    auto argIndices = kernel->get_arg_indices();

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_KERNEL_LAUNCH, command_queue));
      retImplEv->start();
    }

    retError = PI_CHECK_ERROR(cuLaunchKernel(cuFunc, blocksPerGrid[0], 
                                          blocksPerGrid[1], blocksPerGrid[2],
                                          threadsPerBlock[0],
                                          threadsPerBlock[1],
                                          threadsPerBlock[2],
                                          kernel->get_local_size(), cuStream,
                                          argIndices.data(), nullptr));
    kernel->clear_local_size();
    if (event) {
      retError = retImplEv->record();
    }

    if (event) {
      *event = retImplEv.release();
    }
  } catch (pi_result err) {
    retError = err;
  }
  return retError;
}

pi_result
cuda_piEnqueueNativeKernel(pi_queue queue, void (*user_func)(void *), void *args,
                      size_t cb_args, pi_uint32 num_mem_objects,
                      const pi_mem *mem_list, const void **args_mem_loc,
                      pi_uint32 num_events_in_wait_list,
                      const pi_event *event_wait_list, pi_event *event) {
  cl::sycl::detail::pi::die("Not implemented in CUDA backend");
  return {};
}

pi_result cuda_piMemImageCreate( // TODO: change interface to return error code
    pi_context context, pi_mem_flags flags, const pi_image_format *image_format,
    const pi_image_desc *image_desc, void *host_ptr, pi_mem *ret_mem) {
  cl::sycl::detail::pi::die("cuda_piMemImageCreate not implemented");
  return {};
}

pi_result cuda_piMemImageGetInfo(pi_mem image, pi_image_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("cuda_piMemImageGetInfo not implemented");
  return {};
}

pi_result cuda_piMemRetain(pi_mem mem) {
  assert(mem != nullptr);
  assert(mem->get_reference_count() > 0);
  mem->increment_reference_count();
  return PI_SUCCESS;
}

//
// Program
//
pi_result cuda_piProgramCreate(pi_context context, const void *il,
                               size_t length, pi_program *res_program) {
  cl::sycl::detail::pi::die("cuda_piProgramCreate not implemented");
  return {};
}

pi_result cuda_piclProgramCreateWithBinary( // TODO: change to return pi_result
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    pi_int32 *binary_status, pi_program *errcode_ret) {
  cl::sycl::detail::pi::die("cuda_piclProgramCreateWithBinary not implemented");
  return {};
}

pi_result cuda_piProgramGetInfo(pi_program program, pi_program_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  assert(program != nullptr);

  switch (param_name) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->get_reference_count());
  case PI_PROGRAM_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->context_);
  case PI_PROGRAM_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  case PI_PROGRAM_INFO_DEVICES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->context_->deviceId_);
  case PI_PROGRAM_INFO_SOURCE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->source_);
  case PI_PROGRAM_INFO_BINARY_SIZES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->sourceLength_);
  case PI_PROGRAM_INFO_BINARIES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->source_);
  case PI_PROGRAM_INFO_KERNEL_NAMES: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "not implemented");
  }
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Program info request not implemented");
  return {};
}

pi_result cuda_piProgramLink( // TODO: change interface to return error code
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_programs,
    const pi_program *input_programs,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data,
    pi_program *ret_program) {
  cl::sycl::detail::pi::die("cuda_piProgramLink not implemented");
  return {};
}

pi_result cuda_piProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  cl::sycl::detail::pi::die("cuda_piProgramCompile not implemented");
  return {};
}

pi_result cuda_piProgramGetBuildInfo(pi_program program, pi_device device,
                                     cl_program_build_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {

  assert(program != nullptr);

  switch (param_name) {
  case PI_PROGRAM_BUILD_INFO_STATUS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->buildStatus_);
  }
  case PI_PROGRAM_BUILD_INFO_OPTIONS:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->buildOptions_.c_str());
  case PI_PROGRAM_BUILD_INFO_LOG:
    return getInfoArray(program->MAX_LOG_SIZE, param_value_size, param_value,
                        param_value_size_ret, program->infoLog_);
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Program Build info request not implemented");
  return {};
}

pi_result cuda_piProgramRetain(pi_program program) {
  assert(program != nullptr);
  assert(program->get_reference_count() > 0);
  program->increment_reference_count();
  return PI_SUCCESS;
}

pi_result cuda_piProgramRelease(pi_program program) {
  assert(program != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  assert(program->get_reference_count() != 0 &&
                "Reference count overflow detected in cuda_piProgramRelease.");

  // decrement ref count. If it is 0, delete the program.
  if (program->decrement_reference_count() == 0) {

    std::unique_ptr<_pi_program> program_ptr{program};

    pi_result result = PI_INVALID_PROGRAM;

    try {
      ScopedContext active(program->get_context());
      auto cuModule = program->get();
      result = PI_CHECK_ERROR(cuModuleUnload(cuModule));
    } catch (...) {
      result = PI_OUT_OF_RESOURCES;
    }

    return result;
  }

  return PI_SUCCESS;
}

pi_result cuda_piKernelGetInfo(
    pi_kernel kernel,
    pi_kernel_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_INFO_FUNCTION_NAME:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->get_name());
    case PI_KERNEL_INFO_NUM_ARGS:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->get_num_args());
    case PI_KERNEL_INFO_REFERENCE_COUNT:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->get_reference_count());
    case PI_KERNEL_INFO_CONTEXT: {
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->get_context());
    }
    case PI_KERNEL_INFO_PROGRAM: {
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->get_program());
    }
    default: {
      PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
    }
  }

  return PI_INVALID_KERNEL;
}

pi_result cuda_piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                    pi_kernel_group_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {

  // here we want to query about a kernel's cuda blocks!

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_GROUP_INFO_SIZE: {
      int max_threads = 0;
      cl::sycl::detail::pi::assertion(cuFuncGetAttribute(&max_threads,
                                       CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                       kernel->get()) == CUDA_SUCCESS);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     size_t(max_threads));
    }
    case PI_KERNEL_COMPILE_GROUP_INFO_SIZE: {
      // Returns the work-group size specified in the kernel source or IL.
      // If the work-group size is not specified in the kernel source or IL,
      // (0, 0, 0) is returned.
      // https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/clGetKernelWorkGroupInfo.html

      // TODO: can we extract the work group size from the PTX?
      size_t group_size[3] = {0, 0, 0};
      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, group_size);
    }
    case PI_KERNEL_LOCAL_MEM_SIZE: {
      // OpenCL LOCAL == CUDA SHARED
      int bytes = 0;
      cl::sycl::detail::pi::assertion(cuFuncGetAttribute(&bytes,
                                       CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                       kernel->get()) == CUDA_SUCCESS);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    default:
      PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }

  return PI_INVALID_KERNEL;
}

pi_result cuda_piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device,
    cl_kernel_sub_group_info param_name, // TODO: untie from OpenCL
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("cuda_piKernelGetSubGroupInfo not implemented");
  return {};
}

pi_result cuda_piKernelRetain(pi_kernel kernel) {
  assert(kernel != nullptr);
  assert(kernel->get_reference_count() > 0u);

  kernel->increment_reference_count();
  return PI_SUCCESS;
}

pi_result cuda_piKernelRelease(pi_kernel kernel) {
  assert(kernel != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  assert(kernel->get_reference_count() != 0 &&
                "Reference count overflow detected in cuda_piKernelRelease.");

  // decrement ref count. If it is 0, delete the program.
  if (kernel->decrement_reference_count() == 0) {
    // no internal cuda resources to clean up. Just delete it.
    delete kernel;
    return PI_SUCCESS;
  }

  return PI_SUCCESS;
}

// A NOP for the CUDA backend
pi_result cuda_piKernelSetExecInfo(
    pi_kernel kernel, pi_kernel_exec_info param_name, size_t param_value_size,
    const void *param_value) {
  return PI_SUCCESS;
}

//
// Events
//
pi_result cuda_piEventCreate(pi_context context, pi_event *event) {
  assert(context != nullptr);
  assert(event != nullptr);
  pi_result retErr = PI_SUCCESS;
  pi_event retEvent = nullptr;

  try {
    retEvent = _pi_event::make_user(context);
    if (retEvent == nullptr) {
      retErr = PI_OUT_OF_HOST_MEMORY;
    }
  } catch (pi_result err) {
    retErr = err;
  } catch (...) {
    retErr = PI_OUT_OF_RESOURCES;
  }

  *event = retEvent;
  return retErr;
}

pi_result cuda_piEventGetInfo(pi_event event, pi_event_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert(event != nullptr);

  switch (param_name) {
  case PI_EVENT_INFO_QUEUE:
    return getInfo<pi_queue>(param_value_size, param_value,
                             param_value_size_ret, event->get_queue());
  case PI_EVENT_INFO_COMMAND_TYPE:
    return getInfo<pi_command_type>(param_value_size, param_value,
                                    param_value_size_ret,
                                    event->get_command_type());
  case PI_EVENT_INFO_REFERENCE_COUNT:
    return getInfo<pi_uint32>(param_value_size, param_value,
                              param_value_size_ret,
                              event->get_reference_count());
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    return getInfo<pi_event_status>(param_value_size, param_value,
                                    param_value_size_ret,
                                    event->get_execution_status());
  }
  case PI_EVENT_INFO_CONTEXT:
    return getInfo<pi_context>(param_value_size, param_value,
                               param_value_size_ret, event->get_context());
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_INVALID_EVENT;
}

pi_result cuda_piEventGetProfilingInfo(
    pi_event event,
    cl_profiling_info param_name, // TODO: untie from OpenCL
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {

  assert(event != nullptr);

  // TODO: CUDA only implements elapsed time, PI interface requires changing
  //
  switch (param_name) {
  case CL_PROFILING_COMMAND_START:
    return getInfo<pi_uint64>(param_value_size, param_value,
                              param_value_size_ret, 0);
  case CL_PROFILING_COMMAND_END:
    return getInfo<pi_uint64>(param_value_size, param_value,
                              param_value_size_ret, event->get_end_time());
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Event Profiling info request not implemented");
  return {};
}

pi_result cuda_piEventSetCallback(
    pi_event event, pi_int32 command_exec_callback_type,
    void (*pfn_notify)(pi_event event, pi_int32 event_command_status,
                       void *user_data),
    void *user_data) {
  cl::sycl::detail::pi::die("cuda_piEventSetCallback not implemented");
  return {};
}

pi_result cuda_piEventSetStatus(pi_event event, pi_int32 execution_status) {

  assert(execution_status >= PI_EVENT_COMPLETE &&
         execution_status <= PI_EVENT_QUEUED);

  if (!event || event->is_native_event()) {
    return PI_INVALID_EVENT;
  }

  if (execution_status == PI_EVENT_COMPLETE) {
    return event->set_user_event_complete();
  } else if (execution_status < 0) {
    // TODO: A negative integer value causes all enqueued commands that wait
    // on this user event to be terminated.
    cl::sycl::detail::pi::die("cuda_piEventSetStatus support for negative execution_status not "
                              "implemented.");
  }

  return PI_INVALID_VALUE;
}

pi_result cuda_piEventRetain(pi_event event) {
  assert(event != nullptr);

  const auto refCount = event->increment_reference_count();

  cl::sycl::detail::pi::assertion(
    refCount != 0, "Reference count overflow detected in cuda_piEventRetain.");

  return PI_SUCCESS;
}

pi_result cuda_piEventRelease(pi_event event) {
  assert(event != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  cl::sycl::detail::pi::assertion(
    event->get_reference_count() != 0,
    "Reference count overflow detected in cuda_piEventRelease.");

  // decrement ref count. If it is 0, delete the event.
  if (event->decrement_reference_count() == 0) {
    std::unique_ptr<_pi_event> event_ptr{event};
    pi_result result = PI_INVALID_EVENT;

    if (event->is_native_event()) {
      try {
        ScopedContext active(event->get_context());
        auto cuEvent = event->get();
        result = PI_CHECK_ERROR(cuEventDestroy(cuEvent));
      } catch (...) {
        result = PI_OUT_OF_RESOURCES;
      }
    } else {
      result = PI_SUCCESS;
    }

    return result;
  }

  return PI_SUCCESS;
}

//
// Sampler
//
pi_result cuda_piSamplerCreate(
    pi_context context,
    const cl_sampler_properties *sampler_properties, // TODO: untie from OpenCL
    pi_sampler *result_sampler) {
  cl::sycl::detail::pi::die("cuda_piSamplerCreate not implemented");
  return {};
}

pi_result
cuda_piSamplerGetInfo(pi_sampler sampler,
                      cl_sampler_info param_name, // TODO: untie from OpenCL
                      size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("cuda_piSamplerGetInfo not implemented");
  return {};
}

pi_result cuda_piSamplerRetain(pi_sampler sampler) {
  cl::sycl::detail::pi::die("cuda_piSamplerRetain not implemented");
  return {};
}

pi_result cuda_piSamplerRelease(pi_sampler sampler) {
  cl::sycl::detail::pi::die("cuda_piSamplerRelease not implemented");
  return {};
}

pi_result cuda_piEnqueueEventsWait(pi_queue command_queue,
                                   pi_uint32 num_events_in_wait_list,
                                   const pi_event *event_wait_list,
                                   pi_event *event) {
  if (!command_queue) {
    return PI_INVALID_QUEUE;
  }

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      auto result =
          forEachEvent(event_wait_list, num_events_in_wait_list,
                       [command_queue](pi_event event) -> pi_result {
                         return enqueueEventWait(command_queue, event);
                       });

      if (result != PI_SUCCESS) {
        return result;
      }
    }

    if (event) {
      auto new_event =
          _pi_event::make_native(PI_COMMAND_EVENTS_WAIT, command_queue);
      new_event->start();
      new_event->record();
      *event = new_event;
    }

    return PI_SUCCESS;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

// General 3D memory copy operation
// This function requires the corresponding CUDA context to be at the top of
// the context stack
// If the source and/or destination is on the device, src_ptr and/or dst_ptr
// must be a pointer to a CUdeviceptr
static pi_result commonEnqueueMemBufferCopyRect(
    CUstream cu_stream, const size_t *region, const void *src_ptr,
    const CUmemorytype_enum src_type, const size_t *src_offset,
    size_t src_row_pitch, size_t src_slice_pitch, void *dst_ptr,
    const CUmemorytype_enum dst_type, const size_t *dst_offset,
    size_t dst_row_pitch, size_t dst_slice_pitch) {

  assert(region != nullptr);
  assert(src_offset != nullptr);
  assert(dst_offset != nullptr);

  assert(src_type == CU_MEMORYTYPE_DEVICE || src_type == CU_MEMORYTYPE_HOST);
  assert(dst_type == CU_MEMORYTYPE_DEVICE || dst_type == CU_MEMORYTYPE_HOST);

  src_row_pitch = (!src_row_pitch) ? region[0] : src_row_pitch;
  src_slice_pitch =
      (!src_slice_pitch) ? (region[1] * src_row_pitch) : src_slice_pitch;
  dst_row_pitch = (!dst_row_pitch) ? region[0] : dst_row_pitch;
  dst_slice_pitch =
      (!dst_slice_pitch) ? (region[1] * dst_row_pitch) : dst_slice_pitch;

  CUDA_MEMCPY3D params = {0};

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.srcMemoryType = src_type;
  params.srcDevice = src_type == CU_MEMORYTYPE_DEVICE
                         ? *static_cast<const CUdeviceptr *>(src_ptr)
                         : 0;
  params.srcHost = src_type == CU_MEMORYTYPE_HOST ? src_ptr : nullptr;
  params.srcXInBytes = src_offset[0];
  params.srcY = src_offset[1];
  params.srcZ = src_offset[2];
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstMemoryType = dst_type;
  params.dstDevice = dst_type == CU_MEMORYTYPE_DEVICE
                         ? *static_cast<CUdeviceptr *>(dst_ptr)
                         : 0;
  params.dstHost = dst_type == CU_MEMORYTYPE_HOST ? dst_ptr : nullptr;
  params.dstXInBytes = dst_offset[0];
  params.dstY = dst_offset[1];
  params.dstZ = dst_offset[2];
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;

  return PI_CHECK_ERROR(cuMemcpy3DAsync(&params, cu_stream));
}

pi_result cuda_piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    const size_t *buffer_offset, const size_t *host_offset,
    const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, void *ptr,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *retEvent) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  CUstream cuStream = command_queue->get();
  CUdeviceptr devPtr = buffer->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
        event_wait_list, nullptr);

    if (retEvent) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_READ, command_queue));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        cuStream, region, &devPtr, CU_MEMORYTYPE_DEVICE, buffer_offset,
        buffer_row_pitch, buffer_slice_pitch, ptr, CU_MEMORYTYPE_HOST,
        host_offset, host_row_pitch, host_slice_pitch);

    if (retEvent) {
      retErr = retImplEv->record();
    }

    if (blocking_read) {
      retErr = PI_CHECK_ERROR(cuStreamSynchronize(cuStream));
    }

    if (retEvent) {
      *retEvent = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    const size_t *buffer_offset, const size_t *host_offset,
    const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, const void *ptr,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *retEvent) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  CUstream cuStream = command_queue->get();
  CUdeviceptr devPtr = buffer->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
        event_wait_list, nullptr);

    if (retEvent) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_WRITE, command_queue));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        cuStream, region, ptr, CU_MEMORYTYPE_HOST, host_offset, host_row_pitch,
        host_slice_pitch, &devPtr, CU_MEMORYTYPE_DEVICE, buffer_offset,
        buffer_row_pitch, buffer_slice_pitch);

    if (retEvent) {
      retErr = retImplEv->record();
    }

    if (blocking_write) {
      retErr = PI_CHECK_ERROR(cuStreamSynchronize(cuStream));
    }

    if (retEvent) {
      *retEvent = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                                      pi_mem dst_buffer, size_t src_offset,
                                      size_t dst_offset, size_t size,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  if (!command_queue) {
    return PI_INVALID_QUEUE;
  }

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                          event_wait_list, nullptr);
    }

    pi_result result;

    auto stream = command_queue->get();
    auto src = src_buffer->get() + src_offset;
    auto dst = dst_buffer->get() + dst_offset;

    result = PI_CHECK_ERROR(cuMemcpyDtoDAsync(dst, src, size, stream));

    if (event) {
      auto new_event =
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_COPY, command_queue);
      new_event->record();
      *event = new_event;
    }

    return result;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

pi_result cuda_piEnqueueMemBufferCopyRect(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region,
    size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
    size_t dst_slice_pitch, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  assert(src_buffer != nullptr);
  assert(dst_buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  CUstream cuStream = command_queue->get();
  CUdeviceptr srcPtr = src_buffer->get();
  CUdeviceptr dstPtr = dst_buffer->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                      event_wait_list, nullptr);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_COPY, command_queue));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        cuStream, region, &srcPtr, CU_MEMORYTYPE_DEVICE, src_origin, src_row_pitch,
        src_slice_pitch, &dstPtr, CU_MEMORYTYPE_DEVICE, dst_origin,
        dst_row_pitch, dst_slice_pitch);

    if (event) {
      retImplEv->record();
      *event = retImplEv.release();
    }

  } catch (pi_result err) {
    retErr = err;
  }
  return retErr;
}

pi_result cuda_piEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
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

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      cuda_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                          event_wait_list, nullptr);
    }

    pi_result result;

    auto dstDevice = buffer->get() + offset;
    auto stream = command_queue->get();
    auto N = size / pattern_size;

    // pattern size in bytes
    switch (pattern_size) {
    case 1: {
      auto value = *static_cast<const uint8_t *>(pattern);
      result = PI_CHECK_ERROR(cuMemsetD8Async(dstDevice, value, N, stream));
      break;
    }
    case 2: {
      auto value = *static_cast<const uint16_t *>(pattern);
      result = PI_CHECK_ERROR(cuMemsetD16Async(dstDevice, value, N, stream));
      break;
    }
    case 4: {
      auto value = *static_cast<const uint32_t *>(pattern);
      result = PI_CHECK_ERROR(cuMemsetD32Async(dstDevice, value, N, stream));
      break;
    }
    default: {
      // CUDA has no memset functions that allow setting values more than 4
      // bytes. PI API lets you pass an arbitrary "pattern" to the buffer
      // fill, which can be more than 4 bytes. We must break up the pattern
      // into 4 byte values, and set the buffer using multiple strided calls.
      // This means that one cuMemsetD2D32Async call is made for every 4 bytes
      // in the pattern.

      auto number_of_steps = pattern_size / sizeof(uint32_t);

      // we walk up the pattern in 4-byte steps, and call cuMemset for each
      // 4-byte chunk of the pattern.
      for (auto step = 0u; step < number_of_steps; ++step) {
        // take 4 bytes of the pattern
        auto value = *(static_cast<const uint32_t *>(pattern) + step);

        // offset the pointer to the part of the buffer we want to write to
        auto offset_ptr = dstDevice + (step * sizeof(uint32_t));

        // set all of the pattern chunks
        result = PI_CHECK_ERROR(
            cuMemsetD2D32Async(offset_ptr, pattern_size, value, 1, N, stream));
      }

      break;
    }
    }

    if (event) {
      auto new_event =
          _pi_event::make_native(PI_COMMAND_MEMBUFFER_FILL, command_queue);
      new_event->record();
      *event = new_event;
    }

    return result;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

pi_result cuda_piEnqueueMemImageRead(
    pi_queue command_queue, pi_mem image, pi_bool blocking_read,
    const size_t *origin, const size_t *region, size_t row_pitch,
    size_t slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  cl::sycl::detail::pi::die("cuda_piEnqueueMemImageRead not implemented");
  return {};
}

pi_result
cuda_piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                            pi_bool blocking_write, const size_t *origin,
                            const size_t *region, size_t input_row_pitch,
                            size_t input_slice_pitch, const void *ptr,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {
  cl::sycl::detail::pi::die("cuda_piEnqueueMemImageWrite not implemented");
  return {};
}

pi_result cuda_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
                                     pi_mem dst_image, const size_t *src_origin,
                                     const size_t *dst_origin,
                                     const size_t *region,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  cl::sycl::detail::pi::die("cuda_piEnqueueMemImageCopy not implemented");
  return {};
}

pi_result cuda_piEnqueueMemImageFill(pi_queue command_queue, pi_mem image,
                                     const void *fill_color,
                                     const size_t *origin, const size_t *region,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  cl::sycl::detail::pi::die("cuda_piEnqueueMemImageFill not implemented");
  return {};
}

pi_result cuda_piEnqueueMemBufferMap( 
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_map,
    cl_map_flags map_flags, // TODO: untie from OpenCL
    size_t offset, size_t size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *retEvent, void **ret_map) {

  assert(ret_map != nullptr);

  pi_result ret_err = PI_INVALID_OPERATION;

  // Currently no support for overlapping regions
  if (buffer->get_map_ptr() != nullptr) {
    return ret_err;
  }

  // Allocate a pointer in the host to store the mapped information
  auto hostPtr = buffer->map_to_ptr(offset, map_flags);
  *ret_map = buffer->get_map_ptr();
  if (hostPtr) {
    ret_err = PI_SUCCESS;
  }

  if ((map_flags & CL_MAP_READ) || (map_flags & CL_MAP_WRITE)) {
    ret_err = cuda_piEnqueueMemBufferRead(
        command_queue, buffer, blocking_map, offset, size, hostPtr,
        num_events_in_wait_list, event_wait_list, retEvent);
  }

  return ret_err;
}

pi_result cuda_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
                                 void *mapped_ptr,
                                 pi_uint32 num_events_in_wait_list,
                                 const pi_event *event_wait_list,
                                 pi_event *retEvent) {
  pi_result ret_err = PI_INVALID_OPERATION;

  assert(mapped_ptr != nullptr);
  assert(memobj != nullptr);
  assert(memobj->get_map_ptr() != nullptr);
  assert(memobj->get_map_ptr() == mapped_ptr);

  if ((memobj->get_map_flags() & CL_MAP_WRITE) 
      || (memobj->get_map_flags() & CL_MAP_WRITE_INVALIDATE_REGION)) {
    ret_err = cuda_piEnqueueMemBufferWrite(
      command_queue, memobj, true, memobj->get_map_offset(mapped_ptr),
      memobj->get_size(), mapped_ptr, num_events_in_wait_list, event_wait_list,
      retEvent);
  }

  memobj->unmap(mapped_ptr);
  return ret_err;
}

const char SupportedVersion[] = _PI_H_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  int CompareVersions = strcmp(PluginInit->PiVersion, SupportedVersion);
  if (CompareVersions < 0) {
    // PI interface supports lower version of PI.
    // TODO: Take appropriate actions.
    return PI_INVALID_OPERATION;
  }

  // PI interface supports higher version or the same version.
  strncpy(PluginInit->PluginVersion, SupportedVersion, 4);

// Forward calls to OpenCL RT.
#define _PI_CL(pi_api, cuda_api)                                         \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&cuda_api);

  // Platform
  _PI_CL(piPlatformsGet, cuda_piPlatformsGet)
  _PI_CL(piPlatformGetInfo, cuda_piPlatformGetInfo)
  // Device
  _PI_CL(piDevicesGet, cuda_piDevicesGet)
  _PI_CL(piDeviceGetInfo, cuda_piDeviceGetInfo)
  _PI_CL(piDevicePartition, cuda_piDevicePartition)
  _PI_CL(piDeviceRetain, cuda_piDeviceRetain)
  _PI_CL(piDeviceRelease, cuda_piDeviceRelease)
  _PI_CL(piextDeviceSelectBinary, cuda_piextDeviceSelectBinary)
  _PI_CL(piextGetDeviceFunctionPointer, cuda_piextGetDeviceFunctionPointer)
  // Context
  _PI_CL(piContextCreate, cuda_piContextCreate)
  _PI_CL(piContextGetInfo, cuda_piContextGetInfo)
  _PI_CL(piContextRetain, cuda_piContextRetain)
  _PI_CL(piContextRelease, cuda_piContextRelease)
  // Queue
  _PI_CL(piQueueCreate, cuda_piQueueCreate)
  _PI_CL(piQueueGetInfo, cuda_piQueueGetInfo)
  _PI_CL(piQueueFinish, cuda_piQueueFinish)
  _PI_CL(piQueueRetain, cuda_piQueueRetain)
  _PI_CL(piQueueRelease, cuda_piQueueRelease)
  // Memory
  _PI_CL(piMemBufferCreate, cuda_piMemBufferCreate)
  _PI_CL(piMemImageCreate, cuda_piMemImageCreate)
  _PI_CL(piMemGetInfo, cuda_piMemGetInfo)
  _PI_CL(piMemImageGetInfo, cuda_piMemImageGetInfo)
  _PI_CL(piMemRetain, cuda_piMemRetain)
  _PI_CL(piMemRelease, cuda_piMemRelease)
  _PI_CL(piMemBufferPartition, cuda_piMemBufferPartition)
  // Program
  _PI_CL(piProgramCreate, cuda_piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, cuda_piclProgramCreateWithSource)
  _PI_CL(piclProgramCreateWithBinary, cuda_piclProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, cuda_piProgramGetInfo)
  _PI_CL(piProgramCompile, cuda_piProgramCompile)
  _PI_CL(piProgramBuild, cuda_piProgramBuild)
  _PI_CL(piProgramLink, cuda_piProgramLink)
  _PI_CL(piProgramGetBuildInfo, cuda_piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, cuda_piProgramRetain)
  _PI_CL(piProgramRelease, cuda_piProgramRelease)
  // Kernel
  _PI_CL(piKernelCreate, cuda_piKernelCreate)
  _PI_CL(piKernelSetArg, cuda_piKernelSetArg)
  _PI_CL(piKernelGetInfo, cuda_piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, cuda_piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, cuda_piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, cuda_piKernelRetain)
  _PI_CL(piKernelRelease, cuda_piKernelRelease)
  _PI_CL(piKernelSetExecInfo, cuda_piKernelSetExecInfo)
  // Event
  _PI_CL(piEventCreate, cuda_piEventCreate)
  _PI_CL(piEventGetInfo, cuda_piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, cuda_piEventGetProfilingInfo)
  _PI_CL(piEventsWait, cuda_piEventsWait)
  _PI_CL(piEventSetCallback, cuda_piEventSetCallback)
  _PI_CL(piEventSetStatus, cuda_piEventSetStatus)
  _PI_CL(piEventRetain, cuda_piEventRetain)
  _PI_CL(piEventRelease, cuda_piEventRelease)
  // Sampler
  _PI_CL(piSamplerCreate, cuda_piSamplerCreate)
  _PI_CL(piSamplerGetInfo, cuda_piSamplerGetInfo)
  _PI_CL(piSamplerRetain, cuda_piSamplerRetain)
  _PI_CL(piSamplerRelease, cuda_piSamplerRelease)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, cuda_piEnqueueKernelLaunch)
  _PI_CL(piEnqueueNativeKernel, cuda_piEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, cuda_piEnqueueEventsWait)
  _PI_CL(piEnqueueMemBufferRead, cuda_piEnqueueMemBufferRead)
  _PI_CL(piEnqueueMemBufferReadRect, cuda_piEnqueueMemBufferReadRect)
  _PI_CL(piEnqueueMemBufferWrite, cuda_piEnqueueMemBufferWrite)
  _PI_CL(piEnqueueMemBufferWriteRect, cuda_piEnqueueMemBufferWriteRect)
  _PI_CL(piEnqueueMemBufferCopy, cuda_piEnqueueMemBufferCopy)
  _PI_CL(piEnqueueMemBufferCopyRect, cuda_piEnqueueMemBufferCopyRect)
  _PI_CL(piEnqueueMemBufferFill, cuda_piEnqueueMemBufferFill)
  _PI_CL(piEnqueueMemImageRead, cuda_piEnqueueMemImageRead)
  _PI_CL(piEnqueueMemImageWrite, cuda_piEnqueueMemImageWrite)
  _PI_CL(piEnqueueMemImageCopy, cuda_piEnqueueMemImageCopy)
  _PI_CL(piEnqueueMemImageFill, cuda_piEnqueueMemImageFill)
  _PI_CL(piEnqueueMemBufferMap, cuda_piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, cuda_piEnqueueMemUnmap)
  _PI_CL(piextKernelSetArgMemObj, cuda_piextKernelSetArgMemObj)

#undef _PI_CL

  return PI_SUCCESS;
}

}  // extern "C"


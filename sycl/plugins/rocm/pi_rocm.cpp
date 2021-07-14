//==---------- pi_rocm.cpp - HIP Plugin -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_rocm.cpp
/// Implementation of HIP Plugin.
///
/// \ingroup sycl_pi_rocm

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/hip_definitions.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <pi_rocm.hpp>

#include <algorithm>
#include <cassert>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <string.h>

namespace {
// Hipify doesn't support cuArrayGetDescriptor, on AMD the hipArray can just be
// indexed, but on NVidia it is an opaque type and needs to go through
// cuArrayGetDescriptor so implement a utility function to get the array
// properties
inline void getArrayDesc(hipArray *array, hipArray_Format &format,
                         size_t &channels) {
#if defined(__HIP_PLATFORM_AMD__)
  format = array->Format;
  channels = array->NumChannels;
#elif defined(__HIP_PLATFORM_NVIDIA__)
  CUDA_ARRAY_DESCRIPTOR arrayDesc;
  cuArrayGetDescriptor(&arrayDesc, (CUarray)array);

  format = arrayDesc.Format;
  channels = arrayDesc.NumChannels;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
}

// NVidia HIP headers guard hipArray3DCreate behind __CUDACC__, this does not
// seem to be required and we're not using nvcc to build the HIP PI plugin so
// add the translation function here
#if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__CUDACC__)
inline static hipError_t
hipArray3DCreate(hiparray *pHandle,
                 const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  return hipCUResultTohipError(cuArray3DCreate(pHandle, pAllocateArray));
}
#endif

// hipArray gets turned into cudaArray when using the HIP NVIDIA platform, and
// some CUDA APIs use cudaArray* and others use CUarray, these two represent the
// same type, however when building cudaArray appears as an opaque type, so it
// needs to be explicitly casted to CUarray. In order for this to work for both
// AMD and NVidia we introduce an second hipArray type that will be CUarray for
// NVIDIA and hipArray* for AMD so that we can place the explicit casts when
// necessary for NVIDIA and they will be no-ops for AMD.
#if defined(__HIP_PLATFORM_NVIDIA__)
typedef CUarray hipCUarray;
#elif defined(__HIP_PLATFORM_AMD__)
typedef hipArray *hipCUarray;
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

// Add missing HIP to CUDA defines
#if defined(__HIP_PLATFORM_NVIDIA__)
#define hipMemoryType CUmemorytype
#define hipMemoryTypeHost CU_MEMORYTYPE_HOST
#define hipMemoryTypeDevice CU_MEMORYTYPE_DEVICE
#define hipMemoryTypeArray CU_MEMORYTYPE_ARRAY
#define hipMemoryTypeUnified CU_MEMORYTYPE_UNIFIED
#endif

std::string getHipVersionString() {
  int driver_version = 0;
  if (hipDriverGetVersion(&driver_version) != hipSuccess) {
    return "";
  }
  // The version is returned as (1000 major + 10 minor).
  std::stringstream stream;
  stream << "HIP " << driver_version / 1000 << "."
         << driver_version % 1000 / 10;
  return stream.str();
}

pi_result map_error(hipError_t result) {
  switch (result) {
  case hipSuccess:
    return PI_SUCCESS;
  case hipErrorInvalidContext:
    return PI_INVALID_CONTEXT;
  case hipErrorInvalidDevice:
    return PI_INVALID_DEVICE;
  case hipErrorInvalidValue:
    return PI_INVALID_VALUE;
  case hipErrorOutOfMemory:
    return PI_OUT_OF_HOST_MEMORY;
  case hipErrorLaunchOutOfResources:
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

// Iterates over the event wait list, returns correct pi_result error codes.
// Invokes the callback for the latest event of each queue in the wait list.
// The callback must take a single pi_event argument and return a pi_result.
template <typename Func>
pi_result forLatestEvents(const pi_event *event_wait_list,
                          std::size_t num_events_in_wait_list, Func &&f) {

  if (event_wait_list == nullptr || num_events_in_wait_list == 0) {
    return PI_INVALID_EVENT_WAIT_LIST;
  }

  // Fast path if we only have a single event
  if (num_events_in_wait_list == 1) {
    return f(event_wait_list[0]);
  }

  std::vector<pi_event> events{event_wait_list,
                               event_wait_list + num_events_in_wait_list};
  std::sort(events.begin(), events.end(), [](pi_event e0, pi_event e1) {
    // Tiered sort creating sublists of streams (smallest value first) in which
    // the corresponding events are sorted into a sequence of newest first.
    return e0->get_queue()->stream_ < e1->get_queue()->stream_ ||
           (e0->get_queue()->stream_ == e1->get_queue()->stream_ &&
            e0->get_event_id() > e1->get_event_id());
  });

  bool first = true;
  hipStream_t lastSeenStream = 0;
  for (pi_event event : events) {
    if (!event || (!first && event->get_queue()->stream_ == lastSeenStream)) {
      continue;
    }

    first = false;
    lastSeenStream = event->get_queue()->stream_;

    auto result = f(event);
    if (result != PI_SUCCESS) {
      return result;
    }
  }

  return PI_SUCCESS;
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

  const char *errorString = nullptr;
  const char *errorName = nullptr;
  errorName = hipGetErrorName(result);
  errorString = hipGetErrorString(result);
  std::cerr << "\nPI HIP ERROR:"
            << "\n\tValue:           " << result
            << "\n\tName:            " << errorName
            << "\n\tDescription:     " << errorString
            << "\n\tFunction:        " << function
            << "\n\tSource Location: " << file << ":" << line << "\n"
            << std::endl;

  if (std::getenv("PI_HIP_ABORT") != nullptr) {
    std::abort();
  }

  throw map_error(result);
}

/// \cond NODOXY
#define PI_CHECK_ERROR(result) check_error(result, __func__, __LINE__, __FILE__)

/// RAII type to guarantee recovering original HIP context
/// Scoped context is used across all PI HIP plugin implementation
/// to activate the PI Context on the current thread, matching the
/// HIP driver semantics where the context used for the HIP Driver
/// API is the one active on the thread.
/// The implementation tries to avoid replacing the hipCtx_t if it cans
class ScopedContext {
  pi_context placedContext_;
  hipCtx_t original_;
  bool needToRecover_;

public:
  ScopedContext(pi_context ctxt) : placedContext_{ctxt}, needToRecover_{false} {

    if (!placedContext_) {
      throw PI_INVALID_CONTEXT;
    }

    hipCtx_t desired = placedContext_->get();
    PI_CHECK_ERROR(hipCtxGetCurrent(&original_));
    if (original_ != desired) {
      // Sets the desired context as the active one for the thread
      PI_CHECK_ERROR(hipCtxSetCurrent(desired));
      if (original_ == nullptr) {
        // No context is installed on the current thread
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying HIP context are destroyed. This emulates
        // the behaviour of the HIP runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
      } else {
        needToRecover_ = true;
      }
    }
  }

  ~ScopedContext() {
    if (needToRecover_) {
      PI_CHECK_ERROR(hipCtxSetCurrent(original_));
    }
  }
};

/// \cond NODOXY
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

int getAttribute(pi_device device, hipDeviceAttribute_t attribute) {
  int value;
  cl::sycl::detail::pi::assertion(
      hipDeviceGetAttribute(&value, attribute, device->get()) == hipSuccess);
  return value;
}
/// \endcond

void simpleGuessLocalWorkSize(int *threadsPerBlock,
                              const size_t *global_work_size,
                              const size_t maxThreadsPerBlock[3],
                              pi_kernel kernel) {
  assert(threadsPerBlock != nullptr);
  assert(global_work_size != nullptr);
  assert(kernel != nullptr);
  // int recommendedBlockSize, minGrid;

  // PI_CHECK_ERROR(hipOccupancyMaxPotentialBlockSize(
  //    &minGrid, &recommendedBlockSize, kernel->get(),
  //    0, 0));

  //(void)minGrid; // Not used, avoid warnings

  threadsPerBlock[0] = std::min(static_cast<int>(maxThreadsPerBlock[0]),
                                static_cast<int>(global_work_size[0]));

  // Find a local work group size that is a divisor of the global
  // work group size to produce uniform work groups.
  while (0u != (global_work_size[0] % threadsPerBlock[0])) {
    --threadsPerBlock[0];
  }
}

} // anonymous namespace

/// ------ Error handling, matching OpenCL plugin semantics.
__SYCL_INLINE_NAMESPACE(cl) {
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

//--------------
// PI object implementation

extern "C" {

// Required in a number of functions, so forward declare here
pi_result rocm_piEnqueueEventsWait(pi_queue command_queue,
                                   pi_uint32 num_events_in_wait_list,
                                   const pi_event *event_wait_list,
                                   pi_event *event);
pi_result rocm_piEventRelease(pi_event event);
pi_result rocm_piEventRetain(pi_event event);

} // extern "C"

/// \endcond

_pi_event::_pi_event(pi_command_type type, pi_context context, pi_queue queue)
    : commandType_{type}, refCount_{1}, isCompleted_{false}, isRecorded_{false},
      isStarted_{false}, evEnd_{nullptr}, evStart_{nullptr}, evQueued_{nullptr},
      queue_{queue}, context_{context} {

  assert(type != PI_COMMAND_TYPE_USER);

  bool profilingEnabled = queue_->properties_ & PI_QUEUE_PROFILING_ENABLE;

  PI_CHECK_ERROR(hipEventCreateWithFlags(
      &evEnd_, profilingEnabled ? hipEventDefault : hipEventDisableTiming));

  if (profilingEnabled) {
    PI_CHECK_ERROR(hipEventCreateWithFlags(&evQueued_, hipEventDefault));
    PI_CHECK_ERROR(hipEventCreateWithFlags(&evStart_, hipEventDefault));
  }

  if (queue_ != nullptr) {
    rocm_piQueueRetain(queue_);
  }
  rocm_piContextRetain(context_);
}

_pi_event::~_pi_event() {
  if (queue_ != nullptr) {
    rocm_piQueueRelease(queue_);
  }
  rocm_piContextRelease(context_);
}

pi_result _pi_event::start() {
  assert(!is_started());
  pi_result result;

  try {
    if (queue_->properties_ & PI_QUEUE_PROFILING_ENABLE) {
      // NOTE: This relies on the default stream to be unused.
      result = PI_CHECK_ERROR(hipEventRecord(evQueued_, 0));
      result = PI_CHECK_ERROR(hipEventRecord(evStart_, queue_->get()));
    }
  } catch (pi_result error) {
    result = error;
  }

  isStarted_ = true;
  return result;
}

pi_uint64 _pi_event::get_queued_time() const {
  float miliSeconds = 0.0f;
  assert(is_started());

  PI_CHECK_ERROR(hipEventElapsedTime(&miliSeconds, evStart_, evEnd_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_uint64 _pi_event::get_start_time() const {
  float miliSeconds = 0.0f;
  assert(is_started());

  PI_CHECK_ERROR(
      hipEventElapsedTime(&miliSeconds, context_->evBase_, evStart_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_uint64 _pi_event::get_end_time() const {
  float miliSeconds = 0.0f;
  assert(is_started() && is_recorded());

  PI_CHECK_ERROR(hipEventElapsedTime(&miliSeconds, context_->evBase_, evEnd_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_result _pi_event::record() {

  if (is_recorded() || !is_started()) {
    return PI_INVALID_EVENT;
  }

  pi_result result = PI_INVALID_OPERATION;

  if (!queue_) {
    return PI_INVALID_QUEUE;
  }

  hipStream_t hipStream = queue_->get();

  try {
    eventId_ = queue_->get_next_event_id();
    if (eventId_ == 0) {
      cl::sycl::detail::pi::die(
          "Unrecoverable program state reached in event identifier overflow");
    }
    result = PI_CHECK_ERROR(hipEventRecord(evEnd_, hipStream));
  } catch (pi_result error) {
    result = error;
  }

  if (result == PI_SUCCESS) {
    isRecorded_ = true;
  }

  return result;
}

pi_result _pi_event::wait() {
  pi_result retErr;
  try {
    retErr = PI_CHECK_ERROR(hipEventSynchronize(evEnd_));
    isCompleted_ = true;
  } catch (pi_result error) {
    retErr = error;
  }

  return retErr;
}

pi_result _pi_event::release() {
  assert(queue_ != nullptr);
  PI_CHECK_ERROR(hipEventDestroy(evEnd_));

  if (queue_->properties_ & PI_QUEUE_PROFILING_ENABLE) {
    PI_CHECK_ERROR(hipEventDestroy(evQueued_));
    PI_CHECK_ERROR(hipEventDestroy(evStart_));
  }

  return PI_SUCCESS;
}

// makes all future work submitted to queue wait for all work captured in event.
pi_result enqueueEventWait(pi_queue queue, pi_event event) {
  // for native events, the hipStreamWaitEvent call is used.
  // This makes all future work submitted to stream wait for all
  // work captured in event.
  return PI_CHECK_ERROR(hipStreamWaitEvent(queue->get(), event->get(), 0));
}

_pi_program::_pi_program(pi_context ctxt)
    : module_{nullptr}, binary_{},
      binarySizeInBytes_{0}, refCount_{1}, context_{ctxt} {
  rocm_piContextRetain(context_);
}

_pi_program::~_pi_program() { rocm_piContextRelease(context_); }

pi_result _pi_program::set_binary(const char *source, size_t length) {
  assert((binary_ == nullptr && binarySizeInBytes_ == 0) &&
         "Re-setting program binary data which has already been set");
  binary_ = source;
  binarySizeInBytes_ = length;
  return PI_SUCCESS;
}

pi_result _pi_program::build_program(const char *build_options) {

  this->buildOptions_ = build_options;

  constexpr const unsigned int numberOfOptions = 4u;

  hipJitOption options[numberOfOptions];
  void *optionVals[numberOfOptions];

  // Pass a buffer for info messages
  options[0] = hipJitOptionInfoLogBuffer;
  optionVals[0] = (void *)infoLog_;
  // Pass the size of the info buffer
  options[1] = hipJitOptionInfoLogBufferSizeBytes;
  optionVals[1] = (void *)(long)MAX_LOG_SIZE;
  // Pass a buffer for error message
  options[2] = hipJitOptionErrorLogBuffer;
  optionVals[2] = (void *)errorLog_;
  // Pass the size of the error buffer
  options[3] = hipJitOptionErrorLogBufferSizeBytes;
  optionVals[3] = (void *)(long)MAX_LOG_SIZE;

  auto result = PI_CHECK_ERROR(
      hipModuleLoadDataEx(&module_, static_cast<const void *>(binary_),
                          numberOfOptions, options, optionVals));

  const auto success = (result == PI_SUCCESS);

  buildStatus_ =
      success ? PI_PROGRAM_BUILD_STATUS_SUCCESS : PI_PROGRAM_BUILD_STATUS_ERROR;

  // If no exception, result is correct
  return success ? PI_SUCCESS : PI_BUILD_PROGRAM_FAILURE;
}

/// Finds kernel names by searching for entry points in the PTX source, as the
/// HIP driver API doesn't expose an operation for this.
/// Note: This is currently only being used by the SYCL program class for the
///       has_kernel method, so an alternative would be to move the has_kernel
///       query to PI and use hipModuleGetFunction to check for a kernel.
std::string getKernelNames(pi_program program) {
  std::string source(program->binary_,
                     program->binary_ + program->binarySizeInBytes_);
  std::regex entries_pattern(".entry\\s+([^\\([:s:]]*)");
  std::string names("");
  std::smatch match;
  bool first_match = true;
  while (std::regex_search(source, match, entries_pattern)) {
    assert(match.size() == 2);
    names += first_match ? "" : ";";
    names += match[1]; // Second element is the group.
    source = match.suffix().str();
    first_match = false;
  }
  return names;
}

/// RAII object that calls the reference count release function on the held PI
/// object on destruction.
///
/// The `dismiss` function stops the release from happening on destruction.
template <typename T> class ReleaseGuard {
private:
  T Captive;

  static pi_result callRelease(pi_device Captive) {
    return rocm_piDeviceRelease(Captive);
  }

  static pi_result callRelease(pi_context Captive) {
    return rocm_piContextRelease(Captive);
  }

  static pi_result callRelease(pi_mem Captive) {
    return rocm_piMemRelease(Captive);
  }

  static pi_result callRelease(pi_program Captive) {
    return rocm_piProgramRelease(Captive);
  }

  static pi_result callRelease(pi_kernel Captive) {
    return rocm_piKernelRelease(Captive);
  }

  static pi_result callRelease(pi_queue Captive) {
    return rocm_piQueueRelease(Captive);
  }

  static pi_result callRelease(pi_event Captive) {
    return rocm_piEventRelease(Captive);
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
        // A reported HIP error is either an implementation or an asynchronous
        // HIP error for which it is unclear if the function that reported it
        // succeeded or not. Either way, the state of the program is compromised
        // and likely unrecoverable.
        cl::sycl::detail::pi::die(
            "Unrecoverable program state reached in rocm_piMemRelease");
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

/// Obtains the HIP platform.
/// There is only one HIP platform, and contains all devices on the system.
/// Triggers the HIP Driver initialization (hipInit) the first time, so this
/// must be the first PI API called.
///
pi_result rocm_piPlatformsGet(pi_uint32 num_entries, pi_platform *platforms,
                              pi_uint32 *num_platforms) {

  try {
    static std::once_flag initFlag;
    static pi_uint32 numPlatforms = 1;
    static _pi_platform platformId;

    if (num_entries == 0 and platforms != nullptr) {
      return PI_INVALID_VALUE;
    }
    if (platforms == nullptr and num_platforms == nullptr) {
      return PI_INVALID_VALUE;
    }

    pi_result err = PI_SUCCESS;

    std::call_once(
        initFlag,
        [](pi_result &err) {
          if (hipInit(0) != hipSuccess) {
            numPlatforms = 0;
            return;
          }
          int numDevices = 0;
          err = PI_CHECK_ERROR(hipGetDeviceCount(&numDevices));
          if (numDevices == 0) {
            numPlatforms = 0;
            return;
          }
          try {
            platformId.devices_.reserve(numDevices);
            for (int i = 0; i < numDevices; ++i) {
              hipDevice_t device;
              err = PI_CHECK_ERROR(hipDeviceGet(&device, i));
              platformId.devices_.emplace_back(
                  new _pi_device{device, &platformId});
            }
          } catch (const std::bad_alloc &) {
            // Signal out-of-memory situation
            platformId.devices_.clear();
            err = PI_OUT_OF_HOST_MEMORY;
          } catch (...) {
            // Clear and rethrow to allow retry
            platformId.devices_.clear();
            throw;
          }
        },
        err);

    if (num_platforms != nullptr) {
      *num_platforms = numPlatforms;
    }

    if (platforms != nullptr) {
      *platforms = &platformId;
    }

    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result rocm_piPlatformGetInfo(pi_platform platform,
                                 pi_platform_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  assert(platform != nullptr);

  switch (param_name) {
  case PI_PLATFORM_INFO_NAME:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "AMD ROCM BACKEND");
  case PI_PLATFORM_INFO_VENDOR:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "AMD Corporation");
  case PI_PLATFORM_INFO_PROFILE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "FULL PROFILE");
  case PI_PLATFORM_INFO_VERSION: {
    auto version = getHipVersionString();
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   version.c_str());
  }
  case PI_PLATFORM_INFO_EXTENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Platform info request not implemented");
  return {};
}

/// \param devices List of devices available on the system
/// \param num_devices Number of elements in the list of devices
/// Requesting a non-GPU device triggers an error, all PI HIP devices
/// are GPUs.
///
pi_result rocm_piDevicesGet(pi_platform platform, pi_device_type device_type,
                            pi_uint32 num_entries, pi_device *devices,
                            pi_uint32 *num_devices) {

  pi_result err = PI_SUCCESS;
  const bool askingForDefault = device_type == PI_DEVICE_TYPE_DEFAULT;
  const bool askingForGPU = device_type & PI_DEVICE_TYPE_GPU;
  const bool returnDevices = askingForDefault || askingForGPU;

  size_t numDevices = returnDevices ? platform->devices_.size() : 0;

  try {
    if (num_devices) {
      *num_devices = numDevices;
    }

    if (returnDevices && devices) {
      for (size_t i = 0; i < std::min(size_t(num_entries), numDevices); ++i) {
        devices[i] = platform->devices_[i].get();
      }
    }

    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

/// \return PI_SUCCESS if the function is exehipted successfully
/// HIP devices are always root devices so retain always returns success.
pi_result rocm_piDeviceRetain(pi_device device) { return PI_SUCCESS; }

pi_result rocm_piContextGetInfo(pi_context context, pi_context_info param_name,
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
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_OUT_OF_RESOURCES;
}

pi_result rocm_piContextRetain(pi_context context) {
  assert(context != nullptr);
  assert(context->get_reference_count() > 0);

  context->increment_reference_count();
  return PI_SUCCESS;
}

pi_result rocm_piextContextSetExtendedDeleter(
    pi_context context, pi_context_extended_deleter function, void *user_data) {
  context->set_extended_deleter(function, user_data);
  return PI_SUCCESS;
}

/// Not applicable to HIP, devices cannot be partitioned.
///
pi_result rocm_piDevicePartition(
    pi_device device,
    const cl_device_partition_property *properties, // TODO: untie from OpenCL
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices) {
  return PI_INVALID_OPERATION;
}

/// \return If available, the first binary that is PTX
///
pi_result rocm_piextDeviceSelectBinary(pi_device device,
                                       pi_device_binary *binaries,
                                       pi_uint32 num_binaries,
                                       pi_uint32 *selected_binary) {
  if (!binaries) {
    cl::sycl::detail::pi::die("No list of device images provided");
  }
  if (num_binaries < 1) {
    cl::sycl::detail::pi::die("No binary images in the list");
  }

  // Look for an image for the ROCm target, and return the first one that is
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
  return PI_INVALID_BINARY;
}

pi_result rocm_piextGetDeviceFunctionPointer(pi_device device,
                                             pi_program program,
                                             const char *function_name,
                                             pi_uint64 *function_pointer_ret) {
  cl::sycl::detail::pi::die(
      "rocm_piextGetDeviceFunctionPointer not implemented");
  return {};
}

/// \return PI_SUCCESS always since HIP devices are always root devices.
///
pi_result rocm_piDeviceRelease(pi_device device) { return PI_SUCCESS; }

pi_result rocm_piDeviceGetInfo(pi_device device, pi_device_info param_name,
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
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&compute_units,
                              hipDeviceAttributeMultiprocessorCount,
                              device->get()) == hipSuccess);
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
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&max_x, hipDeviceAttributeMaxBlockDimX,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(max_x >= 0);

    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&max_y, hipDeviceAttributeMaxBlockDimY,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(max_y >= 0);

    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&max_z, hipDeviceAttributeMaxBlockDimZ,
                              device->get()) == hipSuccess);
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
        hipDeviceGetAttribute(&max_work_group_size,
                              hipDeviceAttributeMaxThreadsPerBlock,
                              device->get()) == hipSuccess);

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
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int max_threads = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&max_threads,
                              hipDeviceAttributeMaxThreadsPerBlock,
                              device->get()) == hipSuccess);
    int warpSize = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                              device->get()) == hipSuccess);
    int maxWarps = (max_threads + warpSize - 1) / warpSize;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<uint32_t>(maxWarps));
  }
  case PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // Volta provides independent thread scheduling
    // TODO: Revisit for previous generation GPUs
    int major = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor,
                              device->get()) == hipSuccess);
    bool ifp = (major >= 7);
    return getInfo(param_value_size, param_value, param_value_size_ret, ifp);
  }
  case PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    int warpSize = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                              device->get()) == hipSuccess);
    size_t sizes[1] = {static_cast<size_t>(warpSize)};
    return getInfoArray<size_t>(1, param_value_size, param_value,
                                param_value_size_ret, sizes);
  }
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    int clock_freq = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&clock_freq, hipDeviceAttributeClockRate,
                              device->get()) == hipSuccess);
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
    // CL_DEVICE_TYPE_HIPSTOM.

    size_t global = 0;
    cl::sycl::detail::pi::assertion(hipDeviceTotalMem(&global, device->get()) ==
                                    hipSuccess);

    auto quarter_global = static_cast<pi_uint32>(global / 4u);

    auto max_alloc = std::max(std::min(1024u * 1024u * 1024u, quarter_global),
                              32u * 1024u * 1024u);

    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{max_alloc});
  }
  case PI_DEVICE_INFO_IMAGE_SUPPORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    // This call doesn't match to HIP as it doesn't have images, but instead
    // surfaces and textures. No clear call in the HIP API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return getInfo(param_value_size, param_value, param_value_size_ret, 128u);
  }
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    // This call doesn't match to HIP as it doesn't have images, but instead
    // surfaces and textures. No clear call in the HIP API to determine this,
    // but some searching found as of SM 2.x 128 are supported.
    return getInfo(param_value_size, param_value, param_value_size_ret, 128u);
  }

  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int tex_height = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&tex_height, hipDeviceAttributeMaxTexture2DHeight,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(tex_height >= 0);
    int surf_height = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&surf_height,
                              hipDeviceAttributeMaxTexture2DHeight,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(surf_height >= 0);

    int min = std::min(tex_height, surf_height);

    return getInfo(param_value_size, param_value, param_value_size_ret, min);
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&tex_width, hipDeviceAttributeMaxTexture2DWidth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(tex_width >= 0);
    int surf_width = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&surf_width, hipDeviceAttributeMaxTexture2DWidth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return getInfo(param_value_size, param_value, param_value_size_ret, min);
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    // Take the smaller of maximum surface and maximum texture height.
    int tex_height = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&tex_height, hipDeviceAttributeMaxTexture3DHeight,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(tex_height >= 0);
    int surf_height = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&surf_height,
                              hipDeviceAttributeMaxTexture3DHeight,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(surf_height >= 0);

    int min = std::min(tex_height, surf_height);

    return getInfo(param_value_size, param_value, param_value_size_ret, min);
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&tex_width, hipDeviceAttributeMaxTexture3DWidth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(tex_width >= 0);
    int surf_width = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&surf_width, hipDeviceAttributeMaxTexture3DWidth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return getInfo(param_value_size, param_value, param_value_size_ret, min);
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    // Take the smaller of maximum surface and maximum texture depth.
    int tex_depth = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&tex_depth, hipDeviceAttributeMaxTexture3DDepth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(tex_depth >= 0);
    int surf_depth = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&surf_depth, hipDeviceAttributeMaxTexture3DDepth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(surf_depth >= 0);

    int min = std::min(tex_depth, surf_depth);

    return getInfo(param_value_size, param_value, param_value_size_ret, min);
  }
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    // Take the smaller of maximum surface and maximum texture width.
    int tex_width = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&tex_width, hipDeviceAttributeMaxTexture1DWidth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(tex_width >= 0);
    int surf_width = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&surf_width, hipDeviceAttributeMaxTexture1DWidth,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(surf_width >= 0);

    int min = std::min(tex_width, surf_width);

    return getInfo(param_value_size, param_value, param_value_size_ret, min);
  }
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_MAX_SAMPLERS: {
    // This call is kind of meaningless for rocm, as samplers don't exist.
    // Closest thing is textures, which is 128.
    return getInfo(param_value_size, param_value, param_value_size_ret, 128u);
  }
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // __global__ function parameters are passed to the device via constant
    // memory and are limited to 4 KB.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{4000u});
  }
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    int mem_base_addr_align = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&mem_base_addr_align,
                              hipDeviceAttributeTextureAlignment,
                              device->get()) == hipSuccess);
    // Multiply by 8 as clGetDeviceInfo returns this value in bits
    mem_base_addr_align *= 8;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   mem_base_addr_align);
  }
  case PI_DEVICE_INFO_HALF_FP_CONFIG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG: {
    auto config = PI_FP_DENORM | PI_FP_INF_NAN | PI_FP_ROUND_TO_NEAREST |
                  PI_FP_ROUND_TO_ZERO | PI_FP_ROUND_TO_INF | PI_FP_FMA |
                  PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    auto config = PI_FP_DENORM | PI_FP_INF_NAN | PI_FP_ROUND_TO_NEAREST |
                  PI_FP_ROUND_TO_ZERO | PI_FP_ROUND_TO_INF | PI_FP_FMA;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   CL_READ_WRITE_CACHE);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // The value is dohipmented for all existing GPUs in the HIP programming
    // guidelines, section "H.3.2. Global Memory".
    return getInfo(param_value_size, param_value, param_value_size_ret, 128u);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    int cache_size = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&cache_size, hipDeviceAttributeL2CacheSize,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(cache_size >= 0);
    // The L2 cache is global to the GPU.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(cache_size));
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    size_t bytes = 0;
    // Runtime API has easy access to this value, driver API info is scarse.
    cl::sycl::detail::pi::assertion(hipDeviceTotalMem(&bytes, device->get()) ==
                                    hipSuccess);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{bytes});
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    int constant_memory = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&constant_memory,
                              hipDeviceAttributeTotalConstantMemory,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(constant_memory >= 0);

    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(constant_memory));
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: is there a way to retrieve this from HIP driver API?
    // Hard coded to value returned by clinfo for OpenCL 1.2 HIP | GeForce GTX
    // 1060 3GB
    return getInfo(param_value_size, param_value, param_value_size_ret, 9u);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE: {
    // OpenCL's "local memory" maps most closely to HIP's "shared memory".
    // HIP has its own definition of "local memory", which maps to OpenCL's
    // "private memory".
    int local_mem_size = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&local_mem_size,
                              hipDeviceAttributeMaxSharedMemoryPerBlock,
                              device->get()) == hipSuccess);
    cl::sycl::detail::pi::assertion(local_mem_size >= 0);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64(local_mem_size));
  }
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    int ecc_enabled = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&ecc_enabled, hipDeviceAttributeEccEnabled,
                              device->get()) == hipSuccess);

    cl::sycl::detail::pi::assertion((ecc_enabled == 0) | (ecc_enabled == 1));
    auto result = static_cast<pi_bool>(ecc_enabled);
    return getInfo(param_value_size, param_value, param_value_size_ret, result);
  }
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY: {
    int is_integrated = 0;
    cl::sycl::detail::pi::assertion(
        hipDeviceGetAttribute(&is_integrated, hipDeviceAttributeIntegrated,
                              device->get()) == hipSuccess);

    cl::sycl::detail::pi::assertion((is_integrated == 0) |
                                    (is_integrated == 1));
    auto result = static_cast<pi_bool>(is_integrated);
    return getInfo(param_value_size, param_value, param_value_size_ret, result);
  }
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // Hard coded to value returned by clinfo for OpenCL 1.2 HIP | GeForce GTX
    // 1060 3GB
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1000u});
  }
  case PI_DEVICE_INFO_ENDIAN_LITTLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_COMPILER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_LINKER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
  }
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    auto capability = CL_EXEC_KERNEL;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    auto capability =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
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
                   device->get_platform());
  }
  case PI_DEVICE_INFO_NAME: {
    static constexpr size_t MAX_DEVICE_NAME_LENGTH = 256u;
    char name[MAX_DEVICE_NAME_LENGTH];
    cl::sycl::detail::pi::assertion(
        hipDeviceGetName(name, MAX_DEVICE_NAME_LENGTH, device->get()) ==
        hipSuccess);
    return getInfoArray(strlen(name) + 1, param_value_size, param_value,
                        param_value_size_ret, name);
  }
  case PI_DEVICE_INFO_VENDOR: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "AMD Corporation");
  }
  case PI_DEVICE_INFO_DRIVER_VERSION: {
    auto version = getHipVersionString();
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   version.c_str());
  }
  case PI_DEVICE_INFO_PROFILE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "HIP");
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
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_TRUE);
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

    // Intel USM extensions

  case PI_DEVICE_INFO_USM_HOST_SUPPORT: {
    // from cl_intel_unified_shared_memory: "The host memory access capabilities
    // apply to any host allocation."
    //
    // query if/how the device can access page-locked host memory, possibly
    // through PCIe, using the same pointer as the host
    pi_bitfield value = {};
    // if (getAttribute(device, HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)) {
    // the device shares a unified address space with the host
    if (getAttribute(device, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      value = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS | PI_USM_CONCURRENT_ACCESS |
              PI_USM_CONCURRENT_ATOMIC_ACCESS;
    } else {
      // on GPU architectures with compute capability lower than 6.x, atomic
      // operations from the GPU to CPU memory will not be atomic with respect
      // to CPU initiated atomic operations
      value = PI_USM_ACCESS | PI_USM_CONCURRENT_ACCESS;
    }
    //}
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    pi_bitfield value = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS |
                        PI_USM_CONCURRENT_ACCESS |
                        PI_USM_CONCURRENT_ATOMIC_ACCESS;
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The single device shared memory access capabilities apply to any shared
    // allocation associated with this device."
    //
    // query if/how the device can access managed memory associated to it
    pi_bitfield value = {};
    if (getAttribute(device, hipDeviceAttributeManagedMemory)) {
      // the device can allocate managed memory on this system
      value = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS;
    }
    if (getAttribute(device, hipDeviceAttributeConcurrentManagedAccess)) {
      // the device can coherently access managed memory concurrently with the
      // CPU
      value |= PI_USM_CONCURRENT_ACCESS;
      if (getAttribute(device, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
        // compute capability 6.x introduces operations that are atomic with
        // respect to other CPUs and GPUs in the system
        value |= PI_USM_CONCURRENT_ATOMIC_ACCESS;
      }
    }
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The cross-device shared memory access capabilities apply to any shared
    // allocation associated with this device, or to any shared memory
    // allocation on another device that also supports the same cross-device
    // shared memory access capability."
    //
    // query if/how the device can access managed memory associated to other
    // devices
    pi_bitfield value = {};
    if (getAttribute(device, hipDeviceAttributeManagedMemory)) {
      // the device can allocate managed memory on this system
      value |= PI_USM_ACCESS;
    }
    if (getAttribute(device, hipDeviceAttributeConcurrentManagedAccess)) {
      // all devices with the CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
      // attribute can coherently access managed memory concurrently with the
      // CPU
      value |= PI_USM_CONCURRENT_ACCESS;
    }
    if (getAttribute(device, hipDeviceAttributeComputeCapabilityMajor) >= 6) {
      // compute capability 6.x introduces operations that are atomic with
      // respect to other CPUs and GPUs in the system
      if (value & PI_USM_ACCESS)
        value |= PI_USM_ATOMIC_ACCESS;
      if (value & PI_USM_CONCURRENT_ACCESS)
        value |= PI_USM_CONCURRENT_ATOMIC_ACCESS;
    }
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The shared system memory access capabilities apply to any allocations
    // made by a system allocator, such as malloc or new."
    //
    // query if/how the device can access pageable host memory allocated by the
    // system allocator
    pi_bitfield value = {};
    if (getAttribute(device, hipDeviceAttributePageableMemoryAccess)) {
      // the link between the device and the host does not support native
      // atomic operations
      value = PI_USM_ACCESS | PI_USM_CONCURRENT_ACCESS;
    }
    return getInfo(param_value_size, param_value, param_value_size_ret, value);
  }

    // TODO: Investigate if this information is available on HIP.
  case PI_DEVICE_INFO_PCI_ADDRESS:
  case PI_DEVICE_INFO_GPU_EU_COUNT:
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
  case PI_DEVICE_INFO_GPU_SLICES:
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    return PI_INVALID_VALUE;

  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Device info request not implemented");
  return {};
}

/// Gets the native HIP handle of a PI device object
///
/// \param[in] device The PI device to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI device object.
///
/// \return PI_SUCCESS
pi_result rocm_piextDeviceGetNativeHandle(pi_device device,
                                          pi_native_handle *nativeHandle) {
  *nativeHandle = static_cast<pi_native_handle>(device->get());
  return PI_SUCCESS;
}

/// Created a PI device object from a HIP device handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI device object from.
/// \param[in] platform is the PI platform of the device.
/// \param[out] device Set to the PI device object created from native handle.
///
/// \return TBD
pi_result rocm_piextDeviceCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                 pi_platform platform,
                                                 pi_device *device) {
  cl::sycl::detail::pi::die(
      "Creation of PI device from native handle not implemented");
  return {};
}

/* Context APIs */

/// Create a PI HIP context.
///
/// By default creates a scoped context and keeps the last active HIP context
/// on top of the HIP context stack.
/// With the __SYCL_PI_CONTEXT_PROPERTIES_HIP_PRIMARY key/id and a value of
/// PI_TRUE creates a primary HIP context and activates it on the HIP context
/// stack.
///
/// \param[in] properties 0 terminated array of key/id-value combinations. Can
/// be nullptr. Only accepts property key/id
/// __SYCL_PI_CONTEXT_PROPERTIES_HIP_PRIMARY with a pi_bool value.
/// \param[in] num_devices Number of devices to create the context for.
/// \param[in] devices Devices to create the context for.
/// \param[in] pfn_notify Callback, currently unused.
/// \param[in] user_data User data for callback.
/// \param[out] retcontext Set to created context on success.
///
/// \return PI_SUCCESS on success, otherwise an error return code.
pi_result rocm_piContextCreate(const pi_context_properties *properties,
                               pi_uint32 num_devices, const pi_device *devices,
                               void (*pfn_notify)(const char *errinfo,
                                                  const void *private_info,
                                                  size_t cb, void *user_data),
                               void *user_data, pi_context *retcontext) {

  assert(devices != nullptr);
  // TODO: How to implement context callback?
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  assert(num_devices == 1);
  // Need input context
  assert(retcontext != nullptr);
  pi_result errcode_ret = PI_SUCCESS;

  // Parse properties.
  bool property_rocm_primary = false;
  while (properties && (0 != *properties)) {
    // Consume property ID.
    pi_context_properties id = *properties;
    ++properties;
    // Consume property value.
    pi_context_properties value = *properties;
    ++properties;
    switch (id) {
    case __SYCL_PI_CONTEXT_PROPERTIES_HIP_PRIMARY:
      assert(value == PI_FALSE || value == PI_TRUE);
      property_rocm_primary = static_cast<bool>(value);
      break;
    default:
      // Unknown property.
      assert(!"Unknown piContextCreate property in property list");
      return PI_INVALID_VALUE;
    }
  }

  std::unique_ptr<_pi_context> piContextPtr{nullptr};
  try {
    hipCtx_t current = nullptr;

    if (property_rocm_primary) {
      // Use the HIP primary context and assume that we want to use it
      // immediately as we want to forge context switches.
      hipCtx_t Ctxt;
      errcode_ret =
          PI_CHECK_ERROR(hipDevicePrimaryCtxRetain(&Ctxt, devices[0]->get()));
      piContextPtr = std::unique_ptr<_pi_context>(
          new _pi_context{_pi_context::kind::primary, Ctxt, *devices});
      errcode_ret = PI_CHECK_ERROR(hipCtxPushCurrent(Ctxt));
    } else {
      // Create a scoped context.
      hipCtx_t newContext;
      PI_CHECK_ERROR(hipCtxGetCurrent(&current));
      errcode_ret = PI_CHECK_ERROR(
          hipCtxCreate(&newContext, hipDeviceMapHost, devices[0]->get()));
      piContextPtr = std::unique_ptr<_pi_context>(new _pi_context{
          _pi_context::kind::user_defined, newContext, *devices});
    }

    // Use default stream to record base event counter
    PI_CHECK_ERROR(
        hipEventCreateWithFlags(&piContextPtr->evBase_, hipEventDefault));
    PI_CHECK_ERROR(hipEventRecord(piContextPtr->evBase_, 0));

    // For non-primary scoped contexts keep the last active on top of the stack
    // as `cuCtxCreate` replaces it implicitly otherwise.
    // Primary contexts are kept on top of the stack, so the previous context
    // is not queried and therefore not recovered.
    if (current != nullptr) {
      PI_CHECK_ERROR(hipCtxSetCurrent(current));
    }

    *retcontext = piContextPtr.release();
  } catch (pi_result err) {
    errcode_ret = err;
  } catch (...) {
    errcode_ret = PI_OUT_OF_RESOURCES;
  }
  return errcode_ret;
}

pi_result rocm_piContextRelease(pi_context ctxt) {

  assert(ctxt != nullptr);

  if (ctxt->decrement_reference_count() > 0) {
    return PI_SUCCESS;
  }
  ctxt->invoke_extended_deleters();

  std::unique_ptr<_pi_context> context{ctxt};

  PI_CHECK_ERROR(hipEventDestroy(context->evBase_));

  if (!ctxt->is_primary()) {
    hipCtx_t hipCtxt = ctxt->get();
    hipCtx_t current = nullptr;
    PI_CHECK_ERROR(hipCtxGetCurrent(&current));
    if (hipCtxt != current) {
      PI_CHECK_ERROR(hipCtxPushCurrent(hipCtxt));
    }
    //  hipErrorNotSupported this API
    // PI_CHECK_ERROR(hipCtxSynchronize());
    PI_CHECK_ERROR(hipCtxGetCurrent(&current));
    if (hipCtxt == current) {
      PI_CHECK_ERROR(hipCtxPopCurrent(&current));
    }
    return PI_CHECK_ERROR(hipCtxDestroy(hipCtxt));
  } else {
    // Primary context is not destroyed, but released
    hipDevice_t hipDev = ctxt->get_device()->get();
    hipCtx_t current;
    PI_CHECK_ERROR(hipCtxPopCurrent(&current));
    return PI_CHECK_ERROR(hipDevicePrimaryCtxRelease(hipDev));
  }

  hipCtx_t hipCtxt = ctxt->get();
  return PI_CHECK_ERROR(hipCtxDestroy(hipCtxt));
}

/// Gets the native HIP handle of a PI context object
///
/// \param[in] context The PI context to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI context object.
///
/// \return PI_SUCCESS
pi_result rocm_piextContextGetNativeHandle(pi_context context,
                                           pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(context->get());
  return PI_SUCCESS;
}

/// Created a PI context object from a HIP context handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI context object from.
/// \param[out] context Set to the PI context object created from native handle.
///
/// \return TBD
pi_result rocm_piextContextCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                  pi_uint32 num_devices,
                                                  const pi_device *devices,
                                                  bool ownNativeHandle,
                                                  pi_context *context) {
  cl::sycl::detail::pi::die(
      "Creation of PI context from native handle not implemented");
  return {};
}

/// Creates a PI Memory object using a HIP memory allocation.
/// Can trigger a manual copy depending on the mode.
/// \TODO Implement USE_HOST_PTR using cuHostRegister
///
pi_result rocm_piMemBufferCreate(pi_context context, pi_mem_flags flags,
                                 size_t size, void *host_ptr, pi_mem *ret_mem,
                                 const pi_mem_properties *properties) {
  // Need input memory object
  assert(ret_mem != nullptr);
  assert(properties == nullptr && "no mem properties goes to rocm RT yet");
  // Currently, USE_HOST_PTR is not implemented using host register
  // since this triggers a weird segfault after program ends.
  // Setting this constant to true enables testing that behavior.
  const bool enableUseHostPtr = false;
  const bool performInitialCopy =
      (flags & PI_MEM_FLAGS_HOST_PTR_COPY) ||
      ((flags & PI_MEM_FLAGS_HOST_PTR_USE) && !enableUseHostPtr);
  pi_result retErr = PI_SUCCESS;
  pi_mem retMemObj = nullptr;

  try {
    ScopedContext active(context);
    void *ptr;
    _pi_mem::mem_::buffer_mem_::alloc_mode allocMode =
        _pi_mem::mem_::buffer_mem_::alloc_mode::classic;

    if ((flags & PI_MEM_FLAGS_HOST_PTR_USE) && enableUseHostPtr) {
      retErr = PI_CHECK_ERROR(
          hipHostRegister(host_ptr, size, hipHostRegisterMapped));
      retErr = PI_CHECK_ERROR(hipHostGetDevicePointer(&ptr, host_ptr, 0));
      allocMode = _pi_mem::mem_::buffer_mem_::alloc_mode::use_host_ptr;
    } else if (flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
      retErr = PI_CHECK_ERROR(hipHostMalloc(&host_ptr, size));
      retErr = PI_CHECK_ERROR(hipHostGetDevicePointer(&ptr, host_ptr, 0));
      allocMode = _pi_mem::mem_::buffer_mem_::alloc_mode::alloc_host_ptr;
    } else {
      retErr = PI_CHECK_ERROR(hipMalloc(&ptr, size));
      if (flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
        allocMode = _pi_mem::mem_::buffer_mem_::alloc_mode::copy_in;
      }
    }

    if (retErr == PI_SUCCESS) {
      pi_mem parentBuffer = nullptr;

      auto devPtr =
          reinterpret_cast<_pi_mem::mem_::mem_::buffer_mem_::native_type>(ptr);
      auto piMemObj = std::unique_ptr<_pi_mem>(new _pi_mem{
          context, parentBuffer, allocMode, devPtr, host_ptr, size});
      if (piMemObj != nullptr) {
        retMemObj = piMemObj.release();
        if (performInitialCopy) {
          // Operates on the default stream of the current HIP context.
          retErr = PI_CHECK_ERROR(hipMemcpyHtoD(devPtr, host_ptr, size));
          // Synchronize with default stream implicitly used by cuMemcpyHtoD
          // to make buffer data available on device before any other PI call
          // uses it.
          if (retErr == PI_SUCCESS) {
            hipStream_t defaultStream = 0;
            retErr = PI_CHECK_ERROR(hipStreamSynchronize(defaultStream));
          }
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

/// Decreases the reference count of the Mem object.
/// If this is zero, calls the relevant HIP Free function
/// \return PI_SUCCESS unless deallocation error
///
pi_result rocm_piMemRelease(pi_mem memObj) {
  assert((memObj != nullptr) && "PI_INVALID_MEM_OBJECTS");

  pi_result ret = PI_SUCCESS;

  try {

    // Do nothing if there are other references
    if (memObj->decrement_reference_count() > 0) {
      return PI_SUCCESS;
    }

    // make sure memObj is released in case PI_CHECK_ERROR throws
    std::unique_ptr<_pi_mem> uniqueMemObj(memObj);

    if (memObj->is_sub_buffer()) {
      return PI_SUCCESS;
    }

    ScopedContext active(uniqueMemObj->get_context());

    if (memObj->mem_type_ == _pi_mem::mem_type::buffer) {
      switch (uniqueMemObj->mem_.buffer_mem_.allocMode_) {
      case _pi_mem::mem_::buffer_mem_::alloc_mode::copy_in:
      case _pi_mem::mem_::buffer_mem_::alloc_mode::classic:
        ret = PI_CHECK_ERROR(
            hipFree((void *)uniqueMemObj->mem_.buffer_mem_.ptr_));
        break;
      case _pi_mem::mem_::buffer_mem_::alloc_mode::use_host_ptr:
        ret = PI_CHECK_ERROR(
            hipHostUnregister(uniqueMemObj->mem_.buffer_mem_.hostPtr_));
        break;
      case _pi_mem::mem_::buffer_mem_::alloc_mode::alloc_host_ptr:
        ret = PI_CHECK_ERROR(
            hipFreeHost(uniqueMemObj->mem_.buffer_mem_.hostPtr_));
      };
    }

    else if (memObj->mem_type_ == _pi_mem::mem_type::surface) {
      ret = PI_CHECK_ERROR(hipDestroySurfaceObject(
          uniqueMemObj->mem_.surface_mem_.get_surface()));
      auto array = uniqueMemObj->mem_.surface_mem_.get_array();
      ret = PI_CHECK_ERROR(hipFreeArray(array));
    }

  } catch (pi_result err) {
    ret = err;
  } catch (...) {
    ret = PI_OUT_OF_RESOURCES;
  }

  if (ret != PI_SUCCESS) {
    // A reported HIP error is either an implementation or an asynchronous HIP
    // error for which it is unclear if the function that reported it succeeded
    // or not. Either way, the state of the program is compromised and likely
    // unrecoverable.
    cl::sycl::detail::pi::die(
        "Unrecoverable program state reached in rocm_piMemRelease");
  }

  return PI_SUCCESS;
}

/// Implements a buffer partition in the HIP backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply implemented
/// as an offset over an existing HIP allocation.
///
pi_result rocm_piMemBufferPartition(pi_mem parent_buffer, pi_mem_flags flags,
                                    pi_buffer_create_type buffer_create_type,
                                    void *buffer_create_info, pi_mem *memObj) {
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
      *reinterpret_cast<pi_buffer_region>(buffer_create_info);
  assert((bufferRegion.size != 0u) && "PI_INVALID_BUFFER_SIZE");

  assert((bufferRegion.origin <= (bufferRegion.origin + bufferRegion.size)) &&
         "Overflow");
  assert(((bufferRegion.origin + bufferRegion.size) <=
          parent_buffer->mem_.buffer_mem_.get_size()) &&
         "PI_INVALID_BUFFER_SIZE");
  // Retained indirectly due to retaining parent buffer below.
  pi_context context = parent_buffer->context_;
  _pi_mem::mem_::buffer_mem_::alloc_mode allocMode =
      _pi_mem::mem_::buffer_mem_::alloc_mode::classic;

  assert(parent_buffer->mem_.buffer_mem_.ptr_ !=
         _pi_mem::mem_::buffer_mem_::native_type{0});
  _pi_mem::mem_::buffer_mem_::native_type ptr =
      parent_buffer->mem_.buffer_mem_.get_with_offset(bufferRegion.origin);

  void *hostPtr = nullptr;
  if (parent_buffer->mem_.buffer_mem_.hostPtr_) {
    hostPtr = static_cast<char *>(parent_buffer->mem_.buffer_mem_.hostPtr_) +
              bufferRegion.origin;
  }

  ReleaseGuard<pi_mem> releaseGuard(parent_buffer);

  std::unique_ptr<_pi_mem> retMemObj{nullptr};
  try {
    ScopedContext active(context);

    retMemObj = std::unique_ptr<_pi_mem>{new _pi_mem{
        context, parent_buffer, allocMode, ptr, hostPtr, bufferRegion.size}};
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

pi_result rocm_piMemGetInfo(pi_mem memObj, cl_mem_info queriedInfo,
                            size_t expectedQuerySize, void *queryOutput,
                            size_t *writtenQuerySize) {

  cl::sycl::detail::pi::die("rocm_piMemGetInfo not implemented");
}

/// Gets the native HIP handle of a PI mem object
///
/// \param[in] mem The PI mem to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI mem object.
///
/// \return PI_SUCCESS

/// Created a PI mem object from a HIP mem handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI mem object from.
/// \param[out] mem Set to the PI mem object created from native handle.
///
/// \return TBD
pi_result rocm_piextMemCreateWithNativeHandle(pi_native_handle nativeHandle,
                                              pi_mem *mem) {
  cl::sycl::detail::pi::die(
      "Creation of PI mem from native handle not implemented");
  return {};
}

/// Creates a `pi_queue` object on the HIP backend.
/// Valid properties
/// * __SYCL_PI_HIP_USE_DEFAULT_STREAM -> hipStreamDefault
/// * __SYCL_PI_HIP_SYNC_WITH_DEFAULT -> hipStreamNonBlocking
/// \return Pi queue object mapping to a HIPStream
///
pi_result rocm_piQueueCreate(pi_context context, pi_device device,
                             pi_queue_properties properties, pi_queue *queue) {
  try {
    pi_result err = PI_SUCCESS;

    std::unique_ptr<_pi_queue> queueImpl{nullptr};

    if (context->get_device() != device) {
      *queue = nullptr;
      return PI_INVALID_DEVICE;
    }

    ScopedContext active(context);

    hipStream_t hipStream;

    err = PI_CHECK_ERROR(hipStreamCreate(&hipStream));
    if (err != PI_SUCCESS) {
      return err;
    }

    queueImpl = std::unique_ptr<_pi_queue>(
        new _pi_queue{hipStream, context, device, properties});

    *queue = queueImpl.release();

    return PI_SUCCESS;
  } catch (pi_result err) {

    return err;

  } catch (...) {

    return PI_OUT_OF_RESOURCES;
  }
}

pi_result rocm_piQueueGetInfo(pi_queue command_queue, pi_queue_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert(command_queue != nullptr);

  switch (param_name) {
  case PI_QUEUE_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->context_);
  case PI_QUEUE_INFO_DEVICE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->device_);
  case PI_QUEUE_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->get_reference_count());
  case PI_QUEUE_INFO_PROPERTIES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->properties_);
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Queue info request not implemented");
  return {};
}

pi_result rocm_piQueueRetain(pi_queue command_queue) {
  assert(command_queue != nullptr);
  assert(command_queue->get_reference_count() > 0);

  command_queue->increment_reference_count();
  return PI_SUCCESS;
}

pi_result rocm_piQueueRelease(pi_queue command_queue) {
  assert(command_queue != nullptr);

  if (command_queue->decrement_reference_count() > 0) {
    return PI_SUCCESS;
  }

  try {
    std::unique_ptr<_pi_queue> queueImpl(command_queue);

    ScopedContext active(command_queue->get_context());

    auto stream = queueImpl->stream_;
    PI_CHECK_ERROR(hipStreamSynchronize(stream));
    PI_CHECK_ERROR(hipStreamDestroy(stream));

    return PI_SUCCESS;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result rocm_piQueueFinish(pi_queue command_queue) {

  // set default result to a negative result (avoid false-positve tests)
  pi_result result = PI_OUT_OF_HOST_MEMORY;

  try {

    assert(command_queue !=
           nullptr); // need PI_ERROR_INVALID_EXTERNAL_HANDLE error code
    ScopedContext active(command_queue->get_context());
    result = PI_CHECK_ERROR(hipStreamSynchronize(command_queue->stream_));

  } catch (pi_result err) {

    result = err;

  } catch (...) {

    result = PI_OUT_OF_RESOURCES;
  }

  return result;
}

/// Gets the native HIP handle of a PI queue object
///
/// \param[in] queue The PI queue to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI queue object.
///
/// \return PI_SUCCESS
pi_result rocm_piextQueueGetNativeHandle(pi_queue queue,
                                         pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(queue->get());
  return PI_SUCCESS;
}

/// Created a PI queue object from a HIP queue handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI queue object from.
/// \param[in] context is the PI context of the queue.
/// \param[out] queue Set to the PI queue object created from native handle.
///
/// \return TBD
pi_result rocm_piextQueueCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                pi_context context,
                                                pi_queue *queue) {
  cl::sycl::detail::pi::die(
      "Creation of PI queue from native handle not implemented");
  return {};
}

pi_result rocm_piEnqueueMemBufferWrite(pi_queue command_queue, pi_mem buffer,
                                       pi_bool blocking_write, size_t offset,
                                       size_t size, void *ptr,
                                       pi_uint32 num_events_in_wait_list,
                                       const pi_event *event_wait_list,
                                       pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);
  pi_result retErr = PI_SUCCESS;
  hipStream_t hipStream = command_queue->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                      event_wait_list, nullptr);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_WRITE, command_queue));
      retImplEv->start();
    }

    retErr = PI_CHECK_ERROR(
        hipMemcpyHtoDAsync(buffer->mem_.buffer_mem_.get_with_offset(offset),
                           ptr, size, hipStream));

    if (event) {
      retErr = retImplEv->record();
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

pi_result rocm_piEnqueueMemBufferRead(pi_queue command_queue, pi_mem buffer,
                                      pi_bool blocking_read, size_t offset,
                                      size_t size, void *ptr,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);
  pi_result retErr = PI_SUCCESS;
  hipStream_t hipStream = command_queue->get();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                      event_wait_list, nullptr);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_READ, command_queue));
      retImplEv->start();
    }

    retErr = PI_CHECK_ERROR(hipMemcpyDtoHAsync(
        ptr, buffer->mem_.buffer_mem_.get_with_offset(offset), size,
        hipStream));

    if (event) {
      retErr = retImplEv->record();
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

pi_result rocm_piEventsWait(pi_uint32 num_events, const pi_event *event_list) {

  try {
    assert(num_events != 0);
    assert(event_list);
    if (num_events == 0) {
      return PI_INVALID_VALUE;
    }

    if (!event_list) {
      return PI_INVALID_EVENT;
    }

    auto context = event_list[0]->get_context();
    ScopedContext active(context);

    auto waitFunc = [context](pi_event event) -> pi_result {
      if (!event) {
        return PI_INVALID_EVENT;
      }

      if (event->get_context() != context) {
        return PI_INVALID_CONTEXT;
      }

      return event->wait();
    };
    return forLatestEvents(event_list, num_events, waitFunc);
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result rocm_piKernelCreate(pi_program program, const char *kernel_name,
                              pi_kernel *kernel) {
  assert(kernel != nullptr);
  assert(program != nullptr);

  pi_result retErr = PI_SUCCESS;
  std::unique_ptr<_pi_kernel> retKernel{nullptr};

  try {
    ScopedContext active(program->get_context());

    hipFunction_t hipFunc;
    retErr = PI_CHECK_ERROR(
        hipModuleGetFunction(&hipFunc, program->get(), kernel_name));

    std::string kernel_name_woffset = std::string(kernel_name) + "_with_offset";
    hipFunction_t hipFuncWithOffsetParam;
    hipError_t offsetRes = hipModuleGetFunction(
        &hipFuncWithOffsetParam, program->get(), kernel_name_woffset.c_str());

    // If there is no kernel with global offset parameter we mark it as missing
    if (offsetRes == hipErrorNotFound) {
      hipFuncWithOffsetParam = nullptr;
    } else {
      retErr = PI_CHECK_ERROR(offsetRes);
    }

    retKernel = std::unique_ptr<_pi_kernel>(
        new _pi_kernel{hipFunc, hipFuncWithOffsetParam, kernel_name, program,
                       program->get_context()});
  } catch (pi_result err) {
    retErr = err;
  } catch (...) {
    retErr = PI_OUT_OF_HOST_MEMORY;
  }

  *kernel = retKernel.release();
  return retErr;
}

pi_result rocm_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
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

pi_result rocm_piextKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                                       const pi_mem *arg_value) {

  assert(kernel != nullptr);
  assert(arg_value != nullptr);

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
        cl::sycl::detail::pi::die(
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

pi_result rocm_piextKernelSetArgSampler(pi_kernel kernel, pi_uint32 arg_index,
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

pi_result rocm_piEnqueueKernelLaunch(
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

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  int threadsPerBlock[3] = {32, 1, 1};
  size_t maxWorkGroupSize = 0u;
  size_t maxThreadsPerBlock[3] = {};
  bool providedLocalWorkGroupSize = (local_work_size != nullptr);

  {
    pi_result retError = rocm_piDeviceGetInfo(
        command_queue->device_, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
        sizeof(maxThreadsPerBlock), maxThreadsPerBlock, nullptr);
    assert(retError == PI_SUCCESS);
    (void)retError;

    retError = rocm_piDeviceGetInfo(
        command_queue->device_, PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
        sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    assert(retError == PI_SUCCESS);
    // The maxWorkGroupsSize = 1024 for AMD GPU
    // The maxThreadsPerBlock = {1024, 1024, 1024}

    if (providedLocalWorkGroupSize) {
      auto isValid = [&](int dim) {
        if (local_work_size[dim] > maxThreadsPerBlock[dim])
          return PI_INVALID_WORK_ITEM_SIZE;
        // Checks that local work sizes are a divisor of the global work sizes
        // which includes that the local work sizes are neither larger than the
        // global work sizes and not 0.
        if (0u == local_work_size[dim])
          return PI_INVALID_WORK_GROUP_SIZE;
        if (0u != (global_work_size[dim] % local_work_size[dim]))
          return PI_INVALID_WORK_GROUP_SIZE;
        threadsPerBlock[dim] = static_cast<int>(local_work_size[dim]);
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

  int blocksPerGrid[3] = {1, 1, 1};

  for (size_t i = 0; i < work_dim; i++) {
    blocksPerGrid[i] =
        static_cast<int>(global_work_size[i] + threadsPerBlock[i] - 1) /
        threadsPerBlock[i];
  }

  pi_result retError = PI_SUCCESS;
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());
    hipStream_t hipStream = command_queue->get();
    hipFunction_t hipFunc = kernel->get();

    retError = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                        event_wait_list, nullptr);

    // Set the implicit global offset parameter if kernel has offset variant
    if (kernel->get_with_offset_parameter()) {
      std::uint32_t rocm_implicit_offset[3] = {0, 0, 0};
      if (global_work_offset) {
        for (size_t i = 0; i < work_dim; i++) {
          rocm_implicit_offset[i] =
              static_cast<std::uint32_t>(global_work_offset[i]);
          if (global_work_offset[i] != 0) {
            hipFunc = kernel->get_with_offset_parameter();
          }
        }
      }
      kernel->set_implicit_offset_arg(sizeof(rocm_implicit_offset),
                                      rocm_implicit_offset);
    }

    auto argIndices = kernel->get_arg_indices();

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_NDRANGE_KERNEL, command_queue));
      retImplEv->start();
    }

    retError = PI_CHECK_ERROR(hipModuleLaunchKernel(
        hipFunc, blocksPerGrid[0], 1, 1, threadsPerBlock[0], 1, 1,
        kernel->get_local_size(), hipStream, argIndices.data(), nullptr));

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

/// \TODO Not implemented
pi_result rocm_piEnqueueNativeKernel(
    pi_queue queue, void (*user_func)(void *), void *args, size_t cb_args,
    pi_uint32 num_mem_objects, const pi_mem *mem_list,
    const void **args_mem_loc, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  cl::sycl::detail::pi::die("Not implemented in HIP backend");
  return {};
}

/// \TODO Not implemented

pi_result rocm_piMemImageCreate(pi_context context, pi_mem_flags flags,
                                const pi_image_format *image_format,
                                const pi_image_desc *image_desc, void *host_ptr,
                                pi_mem *ret_mem) {

  // Need input memory object
  assert(ret_mem != nullptr);
  const bool performInitialCopy = (flags & PI_MEM_FLAGS_HOST_PTR_COPY) ||
                                  ((flags & PI_MEM_FLAGS_HOST_PTR_USE));
  pi_result retErr = PI_SUCCESS;

  // We only support RBGA channel order
  // TODO: check SYCL CTS and spec. May also have to support BGRA
  if (image_format->image_channel_order !=
      pi_image_channel_order::PI_IMAGE_CHANNEL_ORDER_RGBA) {
    cl::sycl::detail::pi::die(
        "rocm_piMemImageCreate only supports RGBA channel order");
  }

  // We have to use cuArray3DCreate, which has some caveats. The height and
  // depth parameters must be set to 0 produce 1D or 2D arrays. image_desc gives
  // a minimum value of 1, so we need to convert the answer.
  HIP_ARRAY3D_DESCRIPTOR array_desc;
  array_desc.NumChannels = 4; // Only support 4 channel image
  array_desc.Flags = 0;       // No flags required
  array_desc.Width = image_desc->image_width;
  if (image_desc->image_type == PI_MEM_TYPE_IMAGE1D) {
    array_desc.Height = 0;
    array_desc.Depth = 0;
  } else if (image_desc->image_type == PI_MEM_TYPE_IMAGE2D) {
    array_desc.Height = image_desc->image_height;
    array_desc.Depth = 0;
  } else if (image_desc->image_type == PI_MEM_TYPE_IMAGE3D) {
    array_desc.Height = image_desc->image_height;
    array_desc.Depth = image_desc->image_depth;
  }

  // We need to get this now in bytes for calculating the total image size later
  size_t pixel_type_size_bytes;

  switch (image_format->image_channel_data_type) {
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8:
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    array_desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
    pixel_type_size_bytes = 1;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    array_desc.Format = HIP_AD_FORMAT_SIGNED_INT8;
    pixel_type_size_bytes = 1;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT16:
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    array_desc.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
    pixel_type_size_bytes = 2;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    array_desc.Format = HIP_AD_FORMAT_SIGNED_INT16;
    pixel_type_size_bytes = 2;
    break;
  case PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    array_desc.Format = HIP_AD_FORMAT_HALF;
    pixel_type_size_bytes = 2;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    array_desc.Format = HIP_AD_FORMAT_UNSIGNED_INT32;
    pixel_type_size_bytes = 4;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    array_desc.Format = HIP_AD_FORMAT_SIGNED_INT32;
    pixel_type_size_bytes = 4;
    break;
  case PI_IMAGE_CHANNEL_TYPE_FLOAT:
    array_desc.Format = HIP_AD_FORMAT_FLOAT;
    pixel_type_size_bytes = 4;
    break;
  default:
    cl::sycl::detail::pi::die(
        "rocm_piMemImageCreate given unsupported image_channel_data_type");
  }

  // When a dimension isn't used image_desc has the size set to 1
  size_t pixel_size_bytes =
      pixel_type_size_bytes * 4; // 4 is the only number of channels we support
  size_t image_size_bytes = pixel_size_bytes * image_desc->image_width *
                            image_desc->image_height * image_desc->image_depth;

  ScopedContext active(context);
  hipArray *image_array;
  retErr = PI_CHECK_ERROR(hipArray3DCreate(
      reinterpret_cast<hipCUarray *>(&image_array), &array_desc));

  try {
    if (performInitialCopy) {
      // We have to use a different copy function for each image dimensionality
      if (image_desc->image_type == PI_MEM_TYPE_IMAGE1D) {
        retErr = PI_CHECK_ERROR(
            hipMemcpyHtoA(image_array, 0, host_ptr, image_size_bytes));
      } else if (image_desc->image_type == PI_MEM_TYPE_IMAGE2D) {
        hip_Memcpy2D cpy_desc;
        memset(&cpy_desc, 0, sizeof(cpy_desc));
        cpy_desc.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
        cpy_desc.srcHost = host_ptr;
        cpy_desc.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
        cpy_desc.dstArray = reinterpret_cast<hipCUarray>(image_array);
        cpy_desc.WidthInBytes = pixel_size_bytes * image_desc->image_width;
        cpy_desc.Height = image_desc->image_height;
        retErr = PI_CHECK_ERROR(hipMemcpyParam2D(&cpy_desc));
      } else if (image_desc->image_type == PI_MEM_TYPE_IMAGE3D) {
        HIP_MEMCPY3D cpy_desc;
        memset(&cpy_desc, 0, sizeof(cpy_desc));
        cpy_desc.srcMemoryType = hipMemoryType::hipMemoryTypeHost;
        cpy_desc.srcHost = host_ptr;
        cpy_desc.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
        cpy_desc.dstArray = reinterpret_cast<hipCUarray>(image_array);
        cpy_desc.WidthInBytes = pixel_size_bytes * image_desc->image_width;
        cpy_desc.Height = image_desc->image_height;
        cpy_desc.Depth = image_desc->image_depth;
        retErr = PI_CHECK_ERROR(hipDrvMemcpy3D(&cpy_desc));
      }
    }

    // HIP_RESOURCE_DESC is a union of different structs, shown here
    // We need to fill it as described here to use it for a surface or texture
    // HIP_RESOURCE_DESC::resType must be HIP_RESOURCE_TYPE_ARRAY and
    // HIP_RESOURCE_DESC::res::array::hArray must be set to a valid HIP array
    // handle.
    // HIP_RESOURCE_DESC::flags must be set to zero

    hipResourceDesc image_res_desc;
    image_res_desc.res.array.array = image_array;
    image_res_desc.resType = hipResourceTypeArray;

    hipSurfaceObject_t surface;
    retErr = PI_CHECK_ERROR(hipCreateSurfaceObject(&surface, &image_res_desc));

    auto piMemObj = std::unique_ptr<_pi_mem>(new _pi_mem{
        context, image_array, surface, image_desc->image_type, host_ptr});

    if (piMemObj == nullptr) {
      return PI_OUT_OF_HOST_MEMORY;
    }

    *ret_mem = piMemObj.release();
  } catch (pi_result err) {
    PI_CHECK_ERROR(hipFreeArray(image_array));
    return err;
  } catch (...) {
    PI_CHECK_ERROR(hipFreeArray(image_array));
    return PI_ERROR_UNKNOWN;
  }
  return retErr;
}

/// \TODO Not implemented
pi_result rocm_piMemImageGetInfo(pi_mem image, pi_image_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("rocm_piMemImageGetInfo not implemented");
  return {};
}

pi_result rocm_piMemRetain(pi_mem mem) {
  assert(mem != nullptr);
  assert(mem->get_reference_count() > 0);
  mem->increment_reference_count();
  return PI_SUCCESS;
}

/// Not used as HIP backend only creates programs from binary.
/// See \ref rocm_piclProgramCreateWithBinary.
///
pi_result rocm_piclProgramCreateWithSource(pi_context context, pi_uint32 count,
                                           const char **strings,
                                           const size_t *lengths,
                                           pi_program *program) {
  cl::sycl::detail::pi::hipPrint(
      "rocm_piclProgramCreateWithSource not implemented");
  return PI_INVALID_OPERATION;
}

/// Loads the images from a PI program into a HIPmodule that can be
/// used later on to extract functions (kernels).
/// See \ref _pi_program for implementation details.
///
pi_result rocm_piProgramBuild(pi_program program, pi_uint32 num_devices,
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

/// \TODO Not implemented
pi_result rocm_piProgramCreate(pi_context context, const void *il,
                               size_t length, pi_program *res_program) {
  cl::sycl::detail::pi::die("rocm_piProgramCreate not implemented");
  return {};
}

/// Loads images from a list of PTX or HIPBIN binaries.
/// Note: No calls to HIP driver API in this function, only store binaries
/// for later.
///
/// Note: Only supports one device
///
pi_result rocm_piProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    pi_int32 *binary_status, pi_program *program) {
  assert(context != nullptr);
  assert(binaries != nullptr);
  assert(program != nullptr);
  assert(device_list != nullptr);
  assert(num_devices == 1 && "HIP contexts are for a single device");
  assert((context->get_device()->get() == device_list[0]->get()) &&
         "Mismatch between devices context and passed context when creating "
         "program from binary");

  pi_result retError = PI_SUCCESS;

  std::unique_ptr<_pi_program> retProgram{new _pi_program{context}};

  const bool has_length = (lengths != nullptr);
  size_t length = has_length
                      ? lengths[0]
                      : strlen(reinterpret_cast<const char *>(binaries[0])) + 1;

  assert(length != 0);

  retProgram->set_binary(reinterpret_cast<const char *>(binaries[0]), length);

  *program = retProgram.release();

  return retError;
}

pi_result rocm_piProgramGetInfo(pi_program program, pi_program_info param_name,
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
                   program->binary_);
  case PI_PROGRAM_INFO_BINARY_SIZES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->binarySizeInBytes_);
  case PI_PROGRAM_INFO_BINARIES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->binary_);
  case PI_PROGRAM_INFO_KERNEL_NAMES: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   getKernelNames(program).c_str());
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Program info request not implemented");
  return {};
}

/// Creates a new program that is the outcome of the compilation of the headers
///  and the program.
/// \TODO Implement asynchronous compilation
///
pi_result rocm_piProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  assert(program != nullptr);
  assert(num_devices == 1 || num_devices == 0);
  assert(device_list != nullptr || num_devices == 0);
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  assert(num_input_headers == 0);
  pi_result retError = PI_SUCCESS;

  try {
    ScopedContext active(program->get_context());

    program->build_program(options);

  } catch (pi_result err) {
    retError = err;
  }
  return retError;
}

pi_result rocm_piProgramGetBuildInfo(pi_program program, pi_device device,
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Program Build info request not implemented");
  return {};
}

pi_result rocm_piProgramRetain(pi_program program) {
  assert(program != nullptr);
  assert(program->get_reference_count() > 0);
  program->increment_reference_count();
  return PI_SUCCESS;
}

/// Decreases the reference count of a pi_program object.
/// When the reference count reaches 0, it unloads the module from
/// the context.
pi_result rocm_piProgramRelease(pi_program program) {
  assert(program != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  assert(program->get_reference_count() != 0 &&
         "Reference count overflow detected in rocm_piProgramRelease.");

  // decrement ref count. If it is 0, delete the program.
  if (program->decrement_reference_count() == 0) {

    std::unique_ptr<_pi_program> program_ptr{program};

    pi_result result = PI_INVALID_PROGRAM;

    try {
      ScopedContext active(program->get_context());
      auto hipModule = program->get();
      result = PI_CHECK_ERROR(hipModuleUnload(hipModule));
    } catch (...) {
      result = PI_OUT_OF_RESOURCES;
    }

    return result;
  }

  return PI_SUCCESS;
}

/// Gets the native HIP handle of a PI program object
///
/// \param[in] program The PI program to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI program object.
///
/// \return TBD
pi_result rocm_piextProgramGetNativeHandle(pi_program program,
                                           pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(program->get());
  return PI_SUCCESS;
}

/// Created a PI program object from a HIP program handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI program object from.
/// \param[in] context The PI context of the program.
/// \param[out] program Set to the PI program object created from native handle.
///
/// \return TBD
pi_result rocm_piextProgramCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                  pi_context context,
                                                  pi_program *program) {
  cl::sycl::detail::pi::die(
      "Creation of PI program from native handle not implemented");
  return {};
}

pi_result rocm_piKernelGetInfo(pi_kernel kernel, pi_kernel_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {

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
    case PI_KERNEL_INFO_ATTRIBUTES: {
      return getInfo(param_value_size, param_value, param_value_size_ret, "");
    }
    default: {
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
    }
  }

  return PI_INVALID_KERNEL;
}

pi_result rocm_piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                    pi_kernel_group_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {

  // here we want to query about a kernel's rocm blocks!

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
      int max_threads = 0;
      cl::sycl::detail::pi::assertion(
          hipFuncGetAttribute(&max_threads,
                              HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                              kernel->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     size_t(max_threads));
    }
    case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
      // Returns the work-group size specified in the kernel source or IL.
      // If the work-group size is not specified in the kernel source or IL,
      // (0, 0, 0) is returned.
      // https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/clGetKernelWorkGroupInfo.html

      // TODO: can we extract the work group size from the PTX?
      size_t group_size[3] = {0, 0, 0};
      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, group_size);
    }
    case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
      // OpenCL LOCAL == HIP SHARED
      int bytes = 0;
      cl::sycl::detail::pi::assertion(
          hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                              kernel->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
      // Work groups should be multiples of the warp size
      int warpSize = 0;
      cl::sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                                device->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<size_t>(warpSize));
    }
    case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
      // OpenCL PRIVATE == HIP LOCAL
      int bytes = 0;
      cl::sycl::detail::pi::assertion(
          hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                              kernel->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    default:
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }

  return PI_INVALID_KERNEL;
}

pi_result rocm_piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device, pi_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  if (kernel != nullptr) {
    switch (param_name) {
    case PI_KERNEL_MAX_SUB_GROUP_SIZE: {
      // Sub-group size is equivalent to warp size
      int warpSize = 0;
      cl::sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                                device->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<uint32_t>(warpSize));
    }
    case PI_KERNEL_MAX_NUM_SUB_GROUPS: {
      // Number of sub-groups = max block size / warp size + possible remainder
      int max_threads = 0;
      cl::sycl::detail::pi::assertion(
          hipFuncGetAttribute(&max_threads,
                              HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                              kernel->get()) == hipSuccess);
      int warpSize = 0;
      rocm_piKernelGetSubGroupInfo(kernel, device, PI_KERNEL_MAX_SUB_GROUP_SIZE,
                                   0, nullptr, sizeof(uint32_t), &warpSize,
                                   nullptr);
      int maxWarps = (max_threads + warpSize - 1) / warpSize;
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<uint32_t>(maxWarps));
    }
    case PI_KERNEL_COMPILE_NUM_SUB_GROUPS: {
      // Return value of 0 => not specified
      // TODO: Revisit if PTX is generated for compile-time work-group sizes
      return getInfo(param_value_size, param_value, param_value_size_ret, 0);
    }
    case PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL: {
      // Return value of 0 => unspecified or "auto" sub-group size
      // Correct for now, since warp size may be read from special register
      // TODO: Return warp size once default is primary sub-group size
      // TODO: Revisit if we can recover [[sub_group_size]] attribute from PTX
      return getInfo(param_value_size, param_value, param_value_size_ret, 0);
    }
    default:
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }
  return PI_INVALID_KERNEL;
}

pi_result rocm_piKernelRetain(pi_kernel kernel) {
  assert(kernel != nullptr);
  assert(kernel->get_reference_count() > 0u);

  kernel->increment_reference_count();
  return PI_SUCCESS;
}

pi_result rocm_piKernelRelease(pi_kernel kernel) {
  assert(kernel != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  assert(kernel->get_reference_count() != 0 &&
         "Reference count overflow detected in rocm_piKernelRelease.");

  // decrement ref count. If it is 0, delete the program.
  if (kernel->decrement_reference_count() == 0) {
    // no internal rocm resources to clean up. Just delete it.
    delete kernel;
    return PI_SUCCESS;
  }

  return PI_SUCCESS;
}

// A NOP for the HIP backend
pi_result rocm_piKernelSetExecInfo(pi_kernel kernel,
                                   pi_kernel_exec_info param_name,
                                   size_t param_value_size,
                                   const void *param_value) {
  return PI_SUCCESS;
}

pi_result rocm_piextKernelSetArgPointer(pi_kernel kernel, pi_uint32 arg_index,
                                        size_t arg_size,
                                        const void *arg_value) {
  kernel->set_kernel_arg(arg_index, arg_size, arg_value);
  return PI_SUCCESS;
}

//
// Events
//
pi_result rocm_piEventCreate(pi_context context, pi_event *event) {
  cl::sycl::detail::pi::die("PI Event Create not implemented in HIP backend");
}

pi_result rocm_piEventGetInfo(pi_event event, pi_event_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert(event != nullptr);

  switch (param_name) {
  case PI_EVENT_INFO_COMMAND_QUEUE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   event->get_queue());
  case PI_EVENT_INFO_COMMAND_TYPE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   event->get_command_type());
  case PI_EVENT_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   event->get_reference_count());
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<pi_event_status>(event->get_execution_status()));
  }
  case PI_EVENT_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   event->get_context());
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_INVALID_EVENT;
}

/// Obtain profiling information from PI HIP events
/// \TODO Untie from OpenCL, timings from HIP are only elapsed time.
pi_result rocm_piEventGetProfilingInfo(pi_event event,
                                       pi_profiling_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {

  assert(event != nullptr);

  pi_queue queue = event->get_queue();
  if (queue == nullptr || !(queue->properties_ & PI_QUEUE_PROFILING_ENABLE)) {
    return PI_PROFILING_INFO_NOT_AVAILABLE;
  }

  switch (param_name) {
  case PI_PROFILING_INFO_COMMAND_QUEUED:
  case PI_PROFILING_INFO_COMMAND_SUBMIT:
    return getInfo<pi_uint64>(param_value_size, param_value,
                              param_value_size_ret, event->get_queued_time());
  case PI_PROFILING_INFO_COMMAND_START:
    return getInfo<pi_uint64>(param_value_size, param_value,
                              param_value_size_ret, event->get_start_time());
  case PI_PROFILING_INFO_COMMAND_END:
    return getInfo<pi_uint64>(param_value_size, param_value,
                              param_value_size_ret, event->get_end_time());
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Event Profiling info request not implemented");
  return {};
}

pi_result rocm_piEventSetCallback(pi_event event,
                                  pi_int32 command_exec_callback_type,
                                  pfn_notify notify, void *user_data) {

  cl::sycl::detail::pi::die("Event Callback not implemented in HIP backend");
  return PI_SUCCESS;
}

pi_result rocm_piEventSetStatus(pi_event event, pi_int32 execution_status) {

  cl::sycl::detail::pi::die("Event Set Status not implemented in HIP backend");
  return PI_INVALID_VALUE;
}

pi_result rocm_piEventRetain(pi_event event) {
  assert(event != nullptr);

  const auto refCount = event->increment_reference_count();

  cl::sycl::detail::pi::assertion(
      refCount != 0,
      "Reference count overflow detected in rocm_piEventRetain.");

  return PI_SUCCESS;
}

pi_result rocm_piEventRelease(pi_event event) {
  assert(event != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  cl::sycl::detail::pi::assertion(
      event->get_reference_count() != 0,
      "Reference count overflow detected in rocm_piEventRelease.");

  // decrement ref count. If it is 0, delete the event.
  if (event->decrement_reference_count() == 0) {
    std::unique_ptr<_pi_event> event_ptr{event};
    pi_result result = PI_INVALID_EVENT;
    try {
      ScopedContext active(event->get_context());
      result = event->release();
    } catch (...) {
      result = PI_OUT_OF_RESOURCES;
    }
    return result;
  }

  return PI_SUCCESS;
}

/// Enqueues a wait on the given CUstream for all events.
/// See \ref enqueueEventWait
///
pi_result rocm_piEnqueueEventsWait(pi_queue command_queue,
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
          forLatestEvents(event_wait_list, num_events_in_wait_list,
                          [command_queue](pi_event event) -> pi_result {
                            return enqueueEventWait(command_queue, event);
                          });

      if (result != PI_SUCCESS) {
        return result;
      }
    }

    if (event) {
      *event = _pi_event::make_native(PI_COMMAND_TYPE_MARKER, command_queue);
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

/// Gets the native HIP handle of a PI event object
///
/// \param[in] event The PI event to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI event object.
///
/// \return PI_SUCCESS on success. PI_INVALID_EVENT if given a user event.
pi_result rocm_piextEventGetNativeHandle(pi_event event,
                                         pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(event->get());
  return PI_SUCCESS;
}

/// Created a PI event object from a HIP event handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI event object from.
/// \param[out] event Set to the PI event object created from native handle.
///
/// \return TBD
pi_result rocm_piextEventCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                pi_event *event) {
  cl::sycl::detail::pi::die(
      "Creation of PI event from native handle not implemented");
  return {};
}

/// Creates a PI sampler object
///
/// \param[in] context The context the sampler is created for.
/// \param[in] sampler_properties The properties for the sampler.
/// \param[out] result_sampler Set to the resulting sampler object.
///
/// \return PI_SUCCESS on success. PI_INVALID_VALUE if given an invalid property
///         or if there is multiple of properties from the same category.
pi_result rocm_piSamplerCreate(pi_context context,
                               const pi_sampler_properties *sampler_properties,
                               pi_sampler *result_sampler) {
  std::unique_ptr<_pi_sampler> retImplSampl{new _pi_sampler(context)};

  bool propSeen[3] = {false, false, false};
  for (size_t i = 0; sampler_properties[i] != 0; i += 2) {
    switch (sampler_properties[i]) {
    case PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS:
      if (propSeen[0]) {
        return PI_INVALID_VALUE;
      }
      propSeen[0] = true;
      retImplSampl->props_ |= sampler_properties[i + 1];
      break;
    case PI_SAMPLER_PROPERTIES_FILTER_MODE:
      if (propSeen[1]) {
        return PI_INVALID_VALUE;
      }
      propSeen[1] = true;
      retImplSampl->props_ |=
          (sampler_properties[i + 1] - PI_SAMPLER_FILTER_MODE_NEAREST) << 1;
      break;
    case PI_SAMPLER_PROPERTIES_ADDRESSING_MODE:
      if (propSeen[2]) {
        return PI_INVALID_VALUE;
      }
      propSeen[2] = true;
      retImplSampl->props_ |=
          (sampler_properties[i + 1] - PI_SAMPLER_ADDRESSING_MODE_NONE) << 2;
      break;
    default:
      return PI_INVALID_VALUE;
    }
  }

  if (!propSeen[0]) {
    retImplSampl->props_ |= CL_TRUE;
  }
  // Default filter mode to CL_FILTER_NEAREST
  if (!propSeen[2]) {
    retImplSampl->props_ |= (CL_ADDRESS_CLAMP % CL_ADDRESS_NONE) << 2;
  }

  *result_sampler = retImplSampl.release();
  return PI_SUCCESS;
}

/// Gets information from a PI sampler object
///
/// \param[in] sampler The sampler to get the information from.
/// \param[in] param_name The name of the information to get.
/// \param[in] param_value_size The size of the param_value.
/// \param[out] param_value Set to information value.
/// \param[out] param_value_size_ret Set to the size of the information value.
///
/// \return PI_SUCCESS on success.
pi_result rocm_piSamplerGetInfo(pi_sampler sampler, cl_sampler_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  assert(sampler != nullptr);

  switch (param_name) {
  case PI_SAMPLER_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   sampler->get_reference_count());
  case PI_SAMPLER_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   sampler->context_);
  case PI_SAMPLER_INFO_NORMALIZED_COORDS: {
    pi_bool norm_coords_prop = static_cast<pi_bool>(sampler->props_ & 0x1);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   norm_coords_prop);
  }
  case PI_SAMPLER_INFO_FILTER_MODE: {
    pi_sampler_filter_mode filter_prop = static_cast<pi_sampler_filter_mode>(
        ((sampler->props_ >> 1) & 0x1) + PI_SAMPLER_FILTER_MODE_NEAREST);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   filter_prop);
  }
  case PI_SAMPLER_INFO_ADDRESSING_MODE: {
    pi_sampler_addressing_mode addressing_prop =
        static_cast<pi_sampler_addressing_mode>(
            (sampler->props_ >> 2) + PI_SAMPLER_ADDRESSING_MODE_NONE);
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   addressing_prop);
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  return {};
}

/// Retains a PI sampler object, incrementing its reference count.
///
/// \param[in] sampler The sampler to increment the reference count of.
///
/// \return PI_SUCCESS.
pi_result rocm_piSamplerRetain(pi_sampler sampler) {
  assert(sampler != nullptr);
  sampler->increment_reference_count();
  return PI_SUCCESS;
}

/// Releases a PI sampler object, decrementing its reference count. If the
/// reference count reaches zero, the sampler object is destroyed.
///
/// \param[in] sampler The sampler to decrement the reference count of.
///
/// \return PI_SUCCESS.
pi_result rocm_piSamplerRelease(pi_sampler sampler) {
  assert(sampler != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  cl::sycl::detail::pi::assertion(
      sampler->get_reference_count() != 0,
      "Reference count overflow detected in rocm_piSamplerRelease.");

  // decrement ref count. If it is 0, delete the sampler.
  if (sampler->decrement_reference_count() == 0) {
    delete sampler;
  }

  return PI_SUCCESS;
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

  HIP_MEMCPY3D params = {0};

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

pi_result rocm_piEnqueueMemBufferReadRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_read,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  hipStream_t hipStream = command_queue->get();
  void *devPtr = buffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                      event_wait_list, nullptr);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT, command_queue));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, &devPtr, hipMemoryTypeDevice, buffer_offset,
        buffer_row_pitch, buffer_slice_pitch, ptr, hipMemoryTypeHost,
        host_offset, host_row_pitch, host_slice_pitch);

    if (event) {
      retErr = retImplEv->record();
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

pi_result rocm_piEnqueueMemBufferWriteRect(
    pi_queue command_queue, pi_mem buffer, pi_bool blocking_write,
    pi_buff_rect_offset buffer_offset, pi_buff_rect_offset host_offset,
    pi_buff_rect_region region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, size_t host_row_pitch, size_t host_slice_pitch,
    const void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  assert(buffer != nullptr);
  assert(command_queue != nullptr);

  pi_result retErr = PI_SUCCESS;
  hipStream_t hipStream = command_queue->get();
  void *devPtr = buffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                      event_wait_list, nullptr);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_WRITE_RECT, command_queue));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, ptr, hipMemoryTypeHost, host_offset, host_row_pitch,
        host_slice_pitch, &devPtr, hipMemoryTypeDevice, buffer_offset,
        buffer_row_pitch, buffer_slice_pitch);

    if (event) {
      retErr = retImplEv->record();
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

pi_result rocm_piEnqueueMemBufferCopy(pi_queue command_queue, pi_mem src_buffer,
                                      pi_mem dst_buffer, size_t src_offset,
                                      size_t dst_offset, size_t size,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  if (!command_queue) {
    return PI_INVALID_QUEUE;
  }

  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                               event_wait_list, nullptr);
    }

    pi_result result;

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_COPY, command_queue));
      result = retImplEv->start();
    }

    auto stream = command_queue->get();
    auto src = src_buffer->mem_.buffer_mem_.get_with_offset(src_offset);
    auto dst = dst_buffer->mem_.buffer_mem_.get_with_offset(dst_offset);

    result = PI_CHECK_ERROR(hipMemcpyDtoDAsync(dst, src, size, stream));

    if (event) {
      result = retImplEv->record();
      *event = retImplEv.release();
    }

    return result;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
}

pi_result rocm_piEnqueueMemBufferCopyRect(
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
  hipStream_t hipStream = command_queue->get();
  void *srcPtr = src_buffer->mem_.buffer_mem_.get_void();
  void *dstPtr = dst_buffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<_pi_event> retImplEv{nullptr};

  try {
    ScopedContext active(command_queue->get_context());

    retErr = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                      event_wait_list, nullptr);

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, command_queue));
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

pi_result rocm_piEnqueueMemBufferFill(pi_queue command_queue, pi_mem buffer,
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

    if (event_wait_list) {
      rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                               event_wait_list, nullptr);
    }

    pi_result result;

    if (event) {
      retImplEv = std::unique_ptr<_pi_event>(_pi_event::make_native(
          PI_COMMAND_TYPE_MEM_BUFFER_FILL, command_queue));
      result = retImplEv->start();
    }

    auto dstDevice = buffer->mem_.buffer_mem_.get_with_offset(offset);
    auto stream = command_queue->get();
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
      result = PI_INVALID_VALUE;
      break;
    }
    }

    if (event) {
      result = retImplEv->record();
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
  cl::sycl::detail::pi::die("Invalid iamge format.");
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

  return PI_INVALID_VALUE;
}

pi_result rocm_piEnqueueMemImageRead(
    pi_queue command_queue, pi_mem image, pi_bool blocking_read,
    const size_t *origin, const size_t *region, size_t row_pitch,
    size_t slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  assert(command_queue != nullptr);
  assert(image != nullptr);
  assert(image->mem_type_ == _pi_mem::mem_type::surface);

  pi_result retErr = PI_SUCCESS;
  hipStream_t hipStream = command_queue->get();

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                               event_wait_list, nullptr);
    }

    hipArray *array = image->mem_.surface_mem_.get_array();

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(array, Format, NumChannels);

    int elementByteSize = imageElementByteSize(Format);

    size_t byteOffsetX = origin[0] * elementByteSize * NumChannels;
    size_t bytesToCopy = elementByteSize * NumChannels * region[0];

    pi_mem_type imgType = image->mem_.surface_mem_.get_image_type();

    size_t adjustedRegion[3] = {bytesToCopy, region[1], region[2]};
    size_t srcOffset[3] = {byteOffsetX, origin[1], origin[2]};

    retErr = commonEnqueueMemImageNDCopy(hipStream, imgType, adjustedRegion,
                                         array, hipMemoryTypeArray, srcOffset,
                                         ptr, hipMemoryTypeHost, nullptr);

    if (retErr != PI_SUCCESS) {
      return retErr;
    }

    if (event) {
      auto new_event =
          _pi_event::make_native(PI_COMMAND_TYPE_IMAGE_READ, command_queue);
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

pi_result
rocm_piEnqueueMemImageWrite(pi_queue command_queue, pi_mem image,
                            pi_bool blocking_write, const size_t *origin,
                            const size_t *region, size_t input_row_pitch,
                            size_t input_slice_pitch, const void *ptr,
                            pi_uint32 num_events_in_wait_list,
                            const pi_event *event_wait_list, pi_event *event) {

  assert(command_queue != nullptr);
  assert(image != nullptr);
  assert(image->mem_type_ == _pi_mem::mem_type::surface);

  pi_result retErr = PI_SUCCESS;
  hipStream_t hipStream = command_queue->get();

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                               event_wait_list, nullptr);
    }

    hipArray *array = image->mem_.surface_mem_.get_array();

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(array, Format, NumChannels);

    int elementByteSize = imageElementByteSize(Format);

    size_t byteOffsetX = origin[0] * elementByteSize * NumChannels;
    size_t bytesToCopy = elementByteSize * NumChannels * region[0];

    pi_mem_type imgType = image->mem_.surface_mem_.get_image_type();

    size_t adjustedRegion[3] = {bytesToCopy, region[1], region[2]};
    size_t dstOffset[3] = {byteOffsetX, origin[1], origin[2]};

    retErr = commonEnqueueMemImageNDCopy(hipStream, imgType, adjustedRegion,
                                         ptr, hipMemoryTypeHost, nullptr, array,
                                         hipMemoryTypeArray, dstOffset);

    if (retErr != PI_SUCCESS) {
      return retErr;
    }

    if (event) {
      auto new_event =
          _pi_event::make_native(PI_COMMAND_TYPE_IMAGE_WRITE, command_queue);
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

pi_result rocm_piEnqueueMemImageCopy(pi_queue command_queue, pi_mem src_image,
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
  hipStream_t hipStream = command_queue->get();

  try {
    ScopedContext active(command_queue->get_context());

    if (event_wait_list) {
      rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                               event_wait_list, nullptr);
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

    pi_mem_type imgType = src_image->mem_.surface_mem_.get_image_type();

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
      auto new_event =
          _pi_event::make_native(PI_COMMAND_TYPE_IMAGE_COPY, command_queue);
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

/// \TODO Not implemented in HIP, requires untie from OpenCL
pi_result rocm_piEnqueueMemImageFill(pi_queue command_queue, pi_mem image,
                                     const void *fill_color,
                                     const size_t *origin, const size_t *region,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event) {
  cl::sycl::detail::pi::die("rocm_piEnqueueMemImageFill not implemented");
  return {};
}

/// Implements mapping on the host using a BufferRead operation.
/// Mapped pointers are stored in the pi_mem object.
/// If the buffer uses pinned host memory a pointer to that memory is returned
/// and no read operation is done.
/// \TODO Untie types from OpenCL
///
pi_result rocm_piEnqueueMemBufferMap(pi_queue command_queue, pi_mem buffer,
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

  pi_result ret_err = PI_INVALID_OPERATION;
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
    ret_err = rocm_piEnqueueMemBufferRead(
        command_queue, buffer, blocking_map, offset, size, hostPtr,
        num_events_in_wait_list, event_wait_list, event);
  } else {
    ScopedContext active(command_queue->get_context());

    if (is_pinned) {
      ret_err = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                         event_wait_list, nullptr);
    }

    if (event) {
      try {
        *event = _pi_event::make_native(PI_COMMAND_TYPE_MEM_BUFFER_MAP,
                                        command_queue);
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
pi_result rocm_piEnqueueMemUnmap(pi_queue command_queue, pi_mem memobj,
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
    ret_err = rocm_piEnqueueMemBufferWrite(
        command_queue, memobj, true,
        memobj->mem_.buffer_mem_.get_map_offset(mapped_ptr),
        memobj->mem_.buffer_mem_.get_size(), mapped_ptr,
        num_events_in_wait_list, event_wait_list, event);
  } else {
    ScopedContext active(command_queue->get_context());

    if (is_pinned) {
      ret_err = rocm_piEnqueueEventsWait(command_queue, num_events_in_wait_list,
                                         event_wait_list, nullptr);
    }

    if (event) {
      try {
        *event = _pi_event::make_native(PI_COMMAND_TYPE_MEM_BUFFER_UNMAP,
                                        command_queue);
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

/// USM: Implements USM Host allocations using HIP Pinned Memory
///
pi_result rocm_piextUSMHostAlloc(void **result_ptr, pi_context context,
                                 pi_usm_mem_properties *properties, size_t size,
                                 pi_uint32 alignment) {
  assert(result_ptr != nullptr);
  assert(context != nullptr);
  assert(properties == nullptr);
  pi_result result = PI_SUCCESS;
  try {
    ScopedContext active(context);
    result = PI_CHECK_ERROR(hipHostMalloc(result_ptr, size));
  } catch (pi_result error) {
    result = error;
  }

  assert(alignment == 0 ||
         (result == PI_SUCCESS &&
          reinterpret_cast<std::uintptr_t>(*result_ptr) % alignment == 0));
  return result;
}

/// USM: Implements USM device allocations using a normal HIP device pointer
///
pi_result rocm_piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                                   pi_device device,
                                   pi_usm_mem_properties *properties,
                                   size_t size, pi_uint32 alignment) {
  assert(result_ptr != nullptr);
  assert(context != nullptr);
  assert(device != nullptr);
  assert(properties == nullptr);
  pi_result result = PI_SUCCESS;
  try {
    ScopedContext active(context);
    result = PI_CHECK_ERROR(hipMalloc(result_ptr, size));
  } catch (pi_result error) {
    result = error;
  }

  assert(alignment == 0 ||
         (result == PI_SUCCESS &&
          reinterpret_cast<std::uintptr_t>(*result_ptr) % alignment == 0));
  return result;
}

/// USM: Implements USM Shared allocations using HIP Managed Memory
///
pi_result rocm_piextUSMSharedAlloc(void **result_ptr, pi_context context,
                                   pi_device device,
                                   pi_usm_mem_properties *properties,
                                   size_t size, pi_uint32 alignment) {
  assert(result_ptr != nullptr);
  assert(context != nullptr);
  assert(device != nullptr);
  assert(properties == nullptr);
  pi_result result = PI_SUCCESS;
  try {
    ScopedContext active(context);
    result =
        PI_CHECK_ERROR(hipMallocManaged(result_ptr, size, hipMemAttachGlobal));
  } catch (pi_result error) {
    result = error;
  }

  assert(alignment == 0 ||
         (result == PI_SUCCESS &&
          reinterpret_cast<std::uintptr_t>(*result_ptr) % alignment == 0));
  return result;
}

/// USM: Frees the given USM pointer associated with the context.
///
pi_result rocm_piextUSMFree(pi_context context, void *ptr) {

  assert(context != nullptr);
  pi_result result = PI_SUCCESS;
  try {
    ScopedContext active(context);
    unsigned int type;
    hipPointerAttribute_t hipPointerAttributeType;
    result =
        PI_CHECK_ERROR(hipPointerGetAttributes(&hipPointerAttributeType, ptr));
    type = hipPointerAttributeType.memoryType;
    assert(type == hipMemoryTypeDevice or type == hipMemoryTypeHost);
    if (type == hipMemoryTypeDevice) {
      result = PI_CHECK_ERROR(hipFree(ptr));
    }
    if (type == hipMemoryTypeHost) {
      result = PI_CHECK_ERROR(hipFreeHost(ptr));
    }
  } catch (pi_result error) {
    result = error;
  }
  return result;
}

pi_result rocm_piextUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                     size_t count,
                                     pi_uint32 num_events_in_waitlist,
                                     const pi_event *events_waitlist,
                                     pi_event *event) {

  assert(queue != nullptr);
  assert(ptr != nullptr);
  hipStream_t hipStream = queue->get();
  pi_result result = PI_SUCCESS;
  std::unique_ptr<_pi_event> event_ptr{nullptr};

  try {
    ScopedContext active(queue->get_context());
    result = rocm_piEnqueueEventsWait(queue, num_events_in_waitlist,
                                      events_waitlist, nullptr);
    if (event) {
      event_ptr = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_TYPE_MEM_BUFFER_FILL, queue));
      event_ptr->start();
    }
    result = PI_CHECK_ERROR(
        hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                         (unsigned char)value & 0xFF, count, hipStream));
    if (event) {
      result = event_ptr->record();
      *event = event_ptr.release();
    }
  } catch (pi_result err) {
    result = err;
  }

  return result;
}

pi_result rocm_piextUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                     void *dst_ptr, const void *src_ptr,
                                     size_t size,
                                     pi_uint32 num_events_in_waitlist,
                                     const pi_event *events_waitlist,
                                     pi_event *event) {

  assert(queue != nullptr);
  assert(dst_ptr != nullptr);
  assert(src_ptr != nullptr);
  hipStream_t hipStream = queue->get();
  pi_result result = PI_SUCCESS;
  std::unique_ptr<_pi_event> event_ptr{nullptr};

  try {
    ScopedContext active(queue->get_context());
    result = rocm_piEnqueueEventsWait(queue, num_events_in_waitlist,
                                      events_waitlist, nullptr);
    if (event) {
      event_ptr = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_TYPE_MEM_BUFFER_COPY, queue));
      event_ptr->start();
    }
    result = PI_CHECK_ERROR(
        hipMemcpyAsync(dst_ptr, src_ptr, size, hipMemcpyDefault, hipStream));
    if (event) {
      result = event_ptr->record();
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

pi_result rocm_piextUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                       size_t size,
                                       pi_usm_migration_flags flags,
                                       pi_uint32 num_events_in_waitlist,
                                       const pi_event *events_waitlist,
                                       pi_event *event) {

  assert(queue != nullptr);
  assert(ptr != nullptr);
  hipStream_t hipStream = queue->get();
  pi_result result = PI_SUCCESS;
  std::unique_ptr<_pi_event> event_ptr{nullptr};

  // TODO implement handling the flags once the expected behaviour
  // of piextUSMEnqueuePrefetch is detailed in the USM extension
  assert(flags == 0u);

  try {
    ScopedContext active(queue->get_context());
    result = rocm_piEnqueueEventsWait(queue, num_events_in_waitlist,
                                      events_waitlist, nullptr);
    if (event) {
      event_ptr = std::unique_ptr<_pi_event>(
          _pi_event::make_native(PI_COMMAND_TYPE_MEM_BUFFER_COPY, queue));
      event_ptr->start();
    }
    result = PI_CHECK_ERROR(hipMemPrefetchAsync(
        ptr, size, queue->get_context()->get_device()->get(), hipStream));
    if (event) {
      result = event_ptr->record();
      *event = event_ptr.release();
    }
  } catch (pi_result err) {
    result = err;
  }

  return result;
}

/// USM: memadvise API to govern behavior of automatic migration mechanisms
pi_result rocm_piextUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                        size_t length, pi_mem_advice advice,
                                        pi_event *event) {

  assert(queue != nullptr);
  assert(ptr != nullptr);
  // TODO implement a mapping to hipMemAdvise once the expected behaviour
  // of piextUSMEnqueueMemAdvise is detailed in the USM extension
  return rocm_piEnqueueEventsWait(queue, 0, nullptr, event);

  return PI_SUCCESS;
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
pi_result rocm_piextUSMGetMemAllocInfo(pi_context context, const void *ptr,
                                       pi_mem_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {

  assert(context != nullptr);
  assert(ptr != nullptr);
  pi_result result = PI_SUCCESS;
  hipPointerAttribute_t hipPointerAttributeType;

  try {
    ScopedContext active(context);
    switch (param_name) {
    case PI_MEM_ALLOC_TYPE: {
      unsigned int value;
      // do not throw if hipPointerGetAttribute returns hipErrorInvalidValue
      hipError_t ret = hipPointerGetAttributes(&hipPointerAttributeType, ptr);
      if (ret == hipErrorInvalidValue) {
        // pointer not known to the HIP subsystem
        return getInfo(param_value_size, param_value, param_value_size_ret,
                       PI_MEM_TYPE_UNKNOWN);
      }
      result = check_error(ret, __func__, __LINE__ - 5, __FILE__);
      value = hipPointerAttributeType.isManaged;
      if (value) {
        // pointer to managed memory
        return getInfo(param_value_size, param_value, param_value_size_ret,
                       PI_MEM_TYPE_SHARED);
      }
      result = PI_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, ptr));
      value = hipPointerAttributeType.memoryType;
      assert(value == hipMemoryTypeDevice or value == hipMemoryTypeHost);
      if (value == hipMemoryTypeDevice) {
        // pointer to device memory
        return getInfo(param_value_size, param_value, param_value_size_ret,
                       PI_MEM_TYPE_DEVICE);
      }
      if (value == hipMemoryTypeHost) {
        // pointer to host memory
        return getInfo(param_value_size, param_value, param_value_size_ret,
                       PI_MEM_TYPE_HOST);
      }
      // should never get here
      __builtin_unreachable();
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     PI_MEM_TYPE_UNKNOWN);
    }
    case PI_MEM_ALLOC_BASE_PTR: {
      return PI_INVALID_VALUE;
    }
    case PI_MEM_ALLOC_SIZE: {
      return PI_INVALID_VALUE;
    }

    case PI_MEM_ALLOC_DEVICE: {
      unsigned int value;
      result = PI_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, ptr));
      auto devicePointer =
          static_cast<int *>(hipPointerAttributeType.devicePointer);
      value = *devicePointer;
      pi_platform platform;
      result = rocm_piPlatformsGet(0, &platform, nullptr);
      pi_device device = platform->devices_[value].get();
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     device);
    }
    }
  } catch (pi_result error) {
    result = error;
  }

  return result;
}

// This API is called by Sycl RT to notify the end of the plugin lifetime.
// TODO: add a global variable lifetime management code here (see
// pi_level_zero.cpp for reference) Currently this is just a NOOP.
pi_result rocm_piTearDown(void *PluginParameter) { return PI_SUCCESS; }

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

  // Set whole function table to zero to make it easier to detect if
  // functions are not set up below.
  std::memset(&(PluginInit->PiFunctionTable), 0,
              sizeof(PluginInit->PiFunctionTable));

// Forward calls to HIP RT.
#define _PI_CL(pi_api, rocm_api)                                               \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&rocm_api);

  // Platform
  _PI_CL(piPlatformsGet, rocm_piPlatformsGet)
  _PI_CL(piPlatformGetInfo, rocm_piPlatformGetInfo)
  // Device
  _PI_CL(piDevicesGet, rocm_piDevicesGet)
  _PI_CL(piDeviceGetInfo, rocm_piDeviceGetInfo)
  _PI_CL(piDevicePartition, rocm_piDevicePartition)
  _PI_CL(piDeviceRetain, rocm_piDeviceRetain)
  _PI_CL(piDeviceRelease, rocm_piDeviceRelease)
  _PI_CL(piextDeviceSelectBinary, rocm_piextDeviceSelectBinary)
  _PI_CL(piextGetDeviceFunctionPointer, rocm_piextGetDeviceFunctionPointer)
  _PI_CL(piextDeviceGetNativeHandle, rocm_piextDeviceGetNativeHandle)
  _PI_CL(piextDeviceCreateWithNativeHandle,
         rocm_piextDeviceCreateWithNativeHandle)
  // Context
  _PI_CL(piextContextSetExtendedDeleter, rocm_piextContextSetExtendedDeleter)
  _PI_CL(piContextCreate, rocm_piContextCreate)
  _PI_CL(piContextGetInfo, rocm_piContextGetInfo)
  _PI_CL(piContextRetain, rocm_piContextRetain)
  _PI_CL(piContextRelease, rocm_piContextRelease)
  _PI_CL(piextContextGetNativeHandle, rocm_piextContextGetNativeHandle)
  _PI_CL(piextContextCreateWithNativeHandle,
         rocm_piextContextCreateWithNativeHandle)
  // Queue
  _PI_CL(piQueueCreate, rocm_piQueueCreate)
  _PI_CL(piQueueGetInfo, rocm_piQueueGetInfo)
  _PI_CL(piQueueFinish, rocm_piQueueFinish)
  _PI_CL(piQueueRetain, rocm_piQueueRetain)
  _PI_CL(piQueueRelease, rocm_piQueueRelease)
  _PI_CL(piextQueueGetNativeHandle, rocm_piextQueueGetNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle,
         rocm_piextQueueCreateWithNativeHandle)
  // Memory
  _PI_CL(piMemBufferCreate, rocm_piMemBufferCreate)
  _PI_CL(piMemImageCreate, rocm_piMemImageCreate)
  _PI_CL(piMemGetInfo, rocm_piMemGetInfo)
  _PI_CL(piMemImageGetInfo, rocm_piMemImageGetInfo)
  _PI_CL(piMemRetain, rocm_piMemRetain)
  _PI_CL(piMemRelease, rocm_piMemRelease)
  _PI_CL(piMemBufferPartition, rocm_piMemBufferPartition)
  //_PI_CL(piextMemGetNativeHandle, rocm_piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, rocm_piextMemCreateWithNativeHandle)
  // Program
  _PI_CL(piProgramCreate, rocm_piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, rocm_piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, rocm_piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, rocm_piProgramGetInfo)
  _PI_CL(piProgramCompile, rocm_piProgramCompile)
  _PI_CL(piProgramBuild, rocm_piProgramBuild)
  _PI_CL(piProgramGetBuildInfo, rocm_piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, rocm_piProgramRetain)
  _PI_CL(piProgramRelease, rocm_piProgramRelease)
  _PI_CL(piextProgramGetNativeHandle, rocm_piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle,
         rocm_piextProgramCreateWithNativeHandle)
  // Kernel
  _PI_CL(piKernelCreate, rocm_piKernelCreate)
  _PI_CL(piKernelSetArg, rocm_piKernelSetArg)
  _PI_CL(piKernelGetInfo, rocm_piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, rocm_piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, rocm_piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, rocm_piKernelRetain)
  _PI_CL(piKernelRelease, rocm_piKernelRelease)
  _PI_CL(piKernelSetExecInfo, rocm_piKernelSetExecInfo)
  _PI_CL(piextKernelSetArgPointer, rocm_piextKernelSetArgPointer)
  // Event
  _PI_CL(piEventCreate, rocm_piEventCreate)
  _PI_CL(piEventGetInfo, rocm_piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, rocm_piEventGetProfilingInfo)
  _PI_CL(piEventsWait, rocm_piEventsWait)
  _PI_CL(piEventSetCallback, rocm_piEventSetCallback)
  _PI_CL(piEventSetStatus, rocm_piEventSetStatus)
  _PI_CL(piEventRetain, rocm_piEventRetain)
  _PI_CL(piEventRelease, rocm_piEventRelease)
  _PI_CL(piextEventGetNativeHandle, rocm_piextEventGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle,
         rocm_piextEventCreateWithNativeHandle)
  // Sampler
  _PI_CL(piSamplerCreate, rocm_piSamplerCreate)
  _PI_CL(piSamplerGetInfo, rocm_piSamplerGetInfo)
  _PI_CL(piSamplerRetain, rocm_piSamplerRetain)
  _PI_CL(piSamplerRelease, rocm_piSamplerRelease)
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, rocm_piEnqueueKernelLaunch)
  _PI_CL(piEnqueueNativeKernel, rocm_piEnqueueNativeKernel)
  _PI_CL(piEnqueueEventsWait, rocm_piEnqueueEventsWait)
  _PI_CL(piEnqueueMemBufferRead, rocm_piEnqueueMemBufferRead)
  _PI_CL(piEnqueueMemBufferReadRect, rocm_piEnqueueMemBufferReadRect)
  _PI_CL(piEnqueueMemBufferWrite, rocm_piEnqueueMemBufferWrite)
  _PI_CL(piEnqueueMemBufferWriteRect, rocm_piEnqueueMemBufferWriteRect)
  _PI_CL(piEnqueueMemBufferCopy, rocm_piEnqueueMemBufferCopy)
  _PI_CL(piEnqueueMemBufferCopyRect, rocm_piEnqueueMemBufferCopyRect)
  _PI_CL(piEnqueueMemBufferFill, rocm_piEnqueueMemBufferFill)
  _PI_CL(piEnqueueMemImageRead, rocm_piEnqueueMemImageRead)
  _PI_CL(piEnqueueMemImageWrite, rocm_piEnqueueMemImageWrite)
  _PI_CL(piEnqueueMemImageCopy, rocm_piEnqueueMemImageCopy)
  _PI_CL(piEnqueueMemImageFill, rocm_piEnqueueMemImageFill)
  _PI_CL(piEnqueueMemBufferMap, rocm_piEnqueueMemBufferMap)
  _PI_CL(piEnqueueMemUnmap, rocm_piEnqueueMemUnmap)
  // USM
  _PI_CL(piextUSMHostAlloc, rocm_piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, rocm_piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, rocm_piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, rocm_piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, rocm_piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, rocm_piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, rocm_piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, rocm_piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMGetMemAllocInfo, rocm_piextUSMGetMemAllocInfo)

  _PI_CL(piextKernelSetArgMemObj, rocm_piextKernelSetArgMemObj)
  _PI_CL(piextKernelSetArgSampler, rocm_piextKernelSetArgSampler)
  _PI_CL(piTearDown, rocm_piTearDown)

#undef _PI_CL

  return PI_SUCCESS;
}

} // extern "C"

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

// Iterates over the event wait list, returns correct pi_result error codes.
// Invokes the callback for the latest event of each queue in the wait list.
// The callback must take a single pi_event argument and return a pi_result.
template <typename Func>
pi_result forLatestEvents(const pi_event *event_wait_list,
                          std::size_t num_events_in_wait_list, Func &&f) {

  if (event_wait_list == nullptr || num_events_in_wait_list == 0) {
    return PI_ERROR_INVALID_EVENT_WAIT_LIST;
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
    return e0->get_stream() < e1->get_stream() ||
           (e0->get_stream() == e1->get_stream() &&
            e0->get_event_id() > e1->get_event_id());
  });

  bool first = true;
  hipStream_t lastSeenStream = 0;
  for (pi_event event : events) {
    if (!event || (!first && event->get_stream() == lastSeenStream)) {
      continue;
    }

    first = false;
    lastSeenStream = event->get_stream();

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

ScopedContext::ScopedContext(pi_context ctxt)
    : placedContext_{ctxt}, needToRecover_{false} {
  if (!placedContext_) {
    throw PI_ERROR_INVALID_CONTEXT;
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

    auto result = forLatestEvents(
        event_wait_list, num_events_in_wait_list,
        [stream](pi_event event) -> pi_result {
          if (event->get_stream() == stream) {
            return PI_SUCCESS;
          } else {
            return PI_CHECK_ERROR(hipStreamWaitEvent(stream, event->get(), 0));
          }
        });

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
pi_result hip_piEventRelease(pi_event event);
pi_result hip_piEventRetain(pi_event event);

} // extern "C"

/// \endcond

void _pi_queue::compute_stream_wait_for_barrier_if_needed(hipStream_t stream,
                                                          pi_uint32 stream_i) {
  if (barrier_event_ && !compute_applied_barrier_[stream_i]) {
    PI_CHECK_ERROR(hipStreamWaitEvent(stream, barrier_event_, 0));
    compute_applied_barrier_[stream_i] = true;
  }
}

void _pi_queue::transfer_stream_wait_for_barrier_if_needed(hipStream_t stream,
                                                           pi_uint32 stream_i) {
  if (barrier_event_ && !transfer_applied_barrier_[stream_i]) {
    PI_CHECK_ERROR(hipStreamWaitEvent(stream, barrier_event_, 0));
    transfer_applied_barrier_[stream_i] = true;
  }
}

hipStream_t _pi_queue::get_next_compute_stream(pi_uint32 *stream_token) {
  pi_uint32 stream_i;
  pi_uint32 token;
  while (true) {
    if (num_compute_streams_ < compute_streams_.size()) {
      // the check above is for performance - so as not to lock mutex every time
      std::lock_guard<std::mutex> guard(compute_stream_mutex_);
      // The second check is done after mutex is locked so other threads can not
      // change num_compute_streams_ after that
      if (num_compute_streams_ < compute_streams_.size()) {
        PI_CHECK_ERROR(hipStreamCreateWithFlags(
            &compute_streams_[num_compute_streams_++], flags_));
      }
    }
    token = compute_stream_idx_++;
    stream_i = token % compute_streams_.size();
    // if a stream has been reused before it was next selected round-robin
    // fashion, we want to delay its next use and instead select another one
    // that is more likely to have completed all the enqueued work.
    if (delay_compute_[stream_i]) {
      delay_compute_[stream_i] = false;
    } else {
      break;
    }
  }
  if (stream_token) {
    *stream_token = token;
  }
  hipStream_t res = compute_streams_[stream_i];
  compute_stream_wait_for_barrier_if_needed(res, stream_i);
  return res;
}

hipStream_t _pi_queue::get_next_compute_stream(
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    _pi_stream_guard &guard, pi_uint32 *stream_token) {
  for (pi_uint32 i = 0; i < num_events_in_wait_list; i++) {
    pi_uint32 token = event_wait_list[i]->get_compute_stream_token();
    if (event_wait_list[i]->get_queue() == this && can_reuse_stream(token)) {
      std::unique_lock<std::mutex> compute_sync_guard(
          compute_stream_sync_mutex_);
      // redo the check after lock to avoid data races on
      // last_sync_compute_streams_
      if (can_reuse_stream(token)) {
        pi_uint32 stream_i = token % delay_compute_.size();
        delay_compute_[stream_i] = true;
        if (stream_token) {
          *stream_token = token;
        }
        guard = _pi_stream_guard{std::move(compute_sync_guard)};
        hipStream_t res = event_wait_list[i]->get_stream();
        compute_stream_wait_for_barrier_if_needed(res, stream_i);
        return res;
      }
    }
  }
  guard = {};
  return get_next_compute_stream(stream_token);
}

hipStream_t _pi_queue::get_next_transfer_stream() {
  if (transfer_streams_.empty()) { // for example in in-order queue
    return get_next_compute_stream();
  }
  if (num_transfer_streams_ < transfer_streams_.size()) {
    // the check above is for performance - so as not to lock mutex every time
    std::lock_guard<std::mutex> guard(transfer_stream_mutex_);
    // The second check is done after mutex is locked so other threads can not
    // change num_transfer_streams_ after that
    if (num_transfer_streams_ < transfer_streams_.size()) {
      PI_CHECK_ERROR(hipStreamCreateWithFlags(
          &transfer_streams_[num_transfer_streams_++], flags_));
    }
  }
  pi_uint32 stream_i = transfer_stream_idx_++ % transfer_streams_.size();
  hipStream_t res = transfer_streams_[stream_i];
  transfer_stream_wait_for_barrier_if_needed(res, stream_i);
  return res;
}

_pi_event::_pi_event(pi_command_type type, pi_context context, pi_queue queue,
                     hipStream_t stream, pi_uint32 stream_token)
    : commandType_{type}, refCount_{1}, hasBeenWaitedOn_{false},
      isRecorded_{false}, isStarted_{false}, streamToken_{stream_token},
      evEnd_{nullptr}, evStart_{nullptr}, evQueued_{nullptr}, queue_{queue},
      stream_{stream}, context_{context} {

  assert(type != PI_COMMAND_TYPE_USER);

  bool profilingEnabled = queue_->properties_ & PI_QUEUE_FLAG_PROFILING_ENABLE;

  PI_CHECK_ERROR(hipEventCreateWithFlags(
      &evEnd_, profilingEnabled ? hipEventDefault : hipEventDisableTiming));

  if (profilingEnabled) {
    PI_CHECK_ERROR(hipEventCreateWithFlags(&evQueued_, hipEventDefault));
    PI_CHECK_ERROR(hipEventCreateWithFlags(&evStart_, hipEventDefault));
  }

  if (queue_ != nullptr) {
    hip_piQueueRetain(queue_);
  }
  pi2ur::piContextRetain(context_);
}

_pi_event::~_pi_event() {
  if (queue_ != nullptr) {
    hip_piQueueRelease(queue_);
  }
  pi2ur::piContextRelease(context_);
}

pi_result _pi_event::start() {
  assert(!is_started());
  pi_result result = PI_SUCCESS;

  try {
    if (queue_->properties_ & PI_QUEUE_FLAG_PROFILING_ENABLE) {
      // NOTE: This relies on the default stream to be unused.
      PI_CHECK_ERROR(hipEventRecord(evQueued_, 0));
      PI_CHECK_ERROR(hipEventRecord(evStart_, queue_->get()));
    }
  } catch (pi_result error) {
    result = error;
  }

  isStarted_ = true;
  return result;
}

bool _pi_event::is_completed() const noexcept {
  if (!isRecorded_) {
    return false;
  }
  if (!hasBeenWaitedOn_) {
    const hipError_t ret = hipEventQuery(evEnd_);
    if (ret != hipSuccess && ret != hipErrorNotReady) {
      PI_CHECK_ERROR(ret);
      return false;
    }
    if (ret == hipErrorNotReady) {
      return false;
    }
  }
  return true;
}

pi_uint64 _pi_event::get_queued_time() const {
  float miliSeconds = 0.0f;
  assert(is_started());

  // hipEventSynchronize waits till the event is ready for call to
  // hipEventElapsedTime.
  PI_CHECK_ERROR(hipEventSynchronize(evStart_));
  PI_CHECK_ERROR(hipEventSynchronize(evEnd_));

  PI_CHECK_ERROR(hipEventElapsedTime(&miliSeconds, evStart_, evEnd_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_uint64 _pi_event::get_start_time() const {
  float miliSeconds = 0.0f;
  assert(is_started());

  // hipEventSynchronize waits till the event is ready for call to
  // hipEventElapsedTime.
  PI_CHECK_ERROR(hipEventSynchronize(_pi_platform::evBase_));
  PI_CHECK_ERROR(hipEventSynchronize(evStart_));

  PI_CHECK_ERROR(
      hipEventElapsedTime(&miliSeconds, _pi_platform::evBase_, evStart_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_uint64 _pi_event::get_end_time() const {
  float miliSeconds = 0.0f;
  assert(is_started() && is_recorded());

  // hipEventSynchronize waits till the event is ready for call to
  // hipEventElapsedTime.
  PI_CHECK_ERROR(hipEventSynchronize(_pi_platform::evBase_));
  PI_CHECK_ERROR(hipEventSynchronize(evEnd_));

  PI_CHECK_ERROR(
      hipEventElapsedTime(&miliSeconds, _pi_platform::evBase_, evEnd_));
  return static_cast<pi_uint64>(miliSeconds * 1.0e6);
}

pi_result _pi_event::record() {

  if (is_recorded() || !is_started()) {
    return PI_ERROR_INVALID_EVENT;
  }

  pi_result result = PI_ERROR_INVALID_OPERATION;

  if (!queue_) {
    return PI_ERROR_INVALID_QUEUE;
  }

  try {
    eventId_ = queue_->get_next_event_id();
    if (eventId_ == 0) {
      sycl::detail::pi::die(
          "Unrecoverable program state reached in event identifier overflow");
    }
    result = PI_CHECK_ERROR(hipEventRecord(evEnd_, stream_));
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
    hasBeenWaitedOn_ = true;
  } catch (pi_result error) {
    retErr = error;
  }

  return retErr;
}

pi_result _pi_event::release() {
  assert(queue_ != nullptr);
  PI_CHECK_ERROR(hipEventDestroy(evEnd_));

  if (queue_->properties_ & PI_QUEUE_FLAG_PROFILING_ENABLE) {
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
  queue->for_each_stream([e = event->get()](hipStream_t s) {
    PI_CHECK_ERROR(hipStreamWaitEvent(s, e, 0));
  });
  return PI_SUCCESS;
}

_pi_program::_pi_program(pi_context ctxt)
    : module_{nullptr}, binary_{},
      binarySizeInBytes_{0}, refCount_{1}, context_{ctxt} {
  pi2ur::piContextRetain(context_);
}

_pi_program::~_pi_program() { pi2ur::piContextRelease(context_); }

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
  return success ? PI_SUCCESS : PI_ERROR_BUILD_PROGRAM_FAILURE;
}

/// Finds kernel names by searching for entry points in the PTX source, as the
/// HIP driver API doesn't expose an operation for this.
/// Note: This is currently only being used by the SYCL program class for the
///       has_kernel method, so an alternative would be to move the has_kernel
///       query to PI and use hipModuleGetFunction to check for a kernel.
std::string getKernelNames(pi_program program) {
  (void)program;
  sycl::detail::pi::die("getKernelNames not implemented");
  return {};
}

/// RAII object that calls the reference count release function on the held PI
/// object on destruction.
///
/// The `dismiss` function stops the release from happening on destruction.
template <typename T> class ReleaseGuard {
private:
  T Captive;

  static pi_result callRelease(pi_device Captive) {
    return pi2ur::piDeviceRelease(Captive);
  }

  static pi_result callRelease(pi_context Captive) {
    return pi2ur::piContextRelease(Captive);
  }

  static pi_result callRelease(pi_mem Captive) {
    return hip_piMemRelease(Captive);
  }

  static pi_result callRelease(pi_program Captive) {
    return hip_piProgramRelease(Captive);
  }

  static pi_result callRelease(pi_kernel Captive) {
    return hip_piKernelRelease(Captive);
  }

  static pi_result callRelease(pi_queue Captive) {
    return hip_piQueueRelease(Captive);
  }

  static pi_result callRelease(pi_event Captive) {
    return hip_piEventRelease(Captive);
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
        sycl::detail::pi::die(
            "Unrecoverable program state reached in hip_piMemRelease");
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

/// Creates a PI Memory object using a HIP memory allocation.
/// Can trigger a manual copy depending on the mode.
/// \TODO Implement USE_HOST_PTR using cuHostRegister
///
pi_result
hip_piMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                      void *host_ptr, pi_mem *ret_mem,
                      [[maybe_unused]] const pi_mem_properties *properties) {
  // Need input memory object
  assert(ret_mem != nullptr);
  assert((properties == nullptr || *properties == 0) &&
         "no mem properties goes to HIP RT yet");
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
        retErr = PI_ERROR_OUT_OF_HOST_MEMORY;
      }
    }
  } catch (pi_result err) {
    retErr = err;
  } catch (...) {
    retErr = PI_ERROR_OUT_OF_RESOURCES;
  }

  *ret_mem = retMemObj;

  return retErr;
}

/// Decreases the reference count of the Mem object.
/// If this is zero, calls the relevant HIP Free function
/// \return PI_SUCCESS unless deallocation error
///
pi_result hip_piMemRelease(pi_mem memObj) {
  assert((memObj != nullptr) && "PI_ERROR_INVALID_MEM_OBJECTS");

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
    ret = PI_ERROR_OUT_OF_RESOURCES;
  }

  if (ret != PI_SUCCESS) {
    // A reported HIP error is either an implementation or an asynchronous HIP
    // error for which it is unclear if the function that reported it succeeded
    // or not. Either way, the state of the program is compromised and likely
    // unrecoverable.
    sycl::detail::pi::die(
        "Unrecoverable program state reached in hip_piMemRelease");
  }

  return PI_SUCCESS;
}

/// Implements a buffer partition in the HIP backend.
/// A buffer partition (or a sub-buffer, in OpenCL terms) is simply implemented
/// as an offset over an existing HIP allocation.
///
pi_result hip_piMemBufferPartition(
    pi_mem parent_buffer, pi_mem_flags flags,
    [[maybe_unused]] pi_buffer_create_type buffer_create_type,
    void *buffer_create_info, pi_mem *memObj) {
  assert((parent_buffer != nullptr) && "PI_ERROR_INVALID_MEM_OBJECT");
  assert(parent_buffer->is_buffer() && "PI_ERROR_INVALID_MEM_OBJECTS");
  assert(!parent_buffer->is_sub_buffer() && "PI_ERROR_INVALID_MEM_OBJECT");

  // Default value for flags means PI_MEM_FLAGS_ACCCESS_RW.
  if (flags == 0) {
    flags = PI_MEM_FLAGS_ACCESS_RW;
  }

  assert((flags == PI_MEM_FLAGS_ACCESS_RW) && "PI_ERROR_INVALID_VALUE");
  assert((buffer_create_type == PI_BUFFER_CREATE_TYPE_REGION) &&
         "PI_ERROR_INVALID_VALUE");
  assert((buffer_create_info != nullptr) && "PI_ERROR_INVALID_VALUE");
  assert(memObj != nullptr);

  const auto bufferRegion =
      *reinterpret_cast<pi_buffer_region>(buffer_create_info);
  assert((bufferRegion.size != 0u) && "PI_ERROR_INVALID_BUFFER_SIZE");

  assert((bufferRegion.origin <= (bufferRegion.origin + bufferRegion.size)) &&
         "Overflow");
  assert(((bufferRegion.origin + bufferRegion.size) <=
          parent_buffer->mem_.buffer_mem_.get_size()) &&
         "PI_ERROR_INVALID_BUFFER_SIZE");
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
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  }

  releaseGuard.dismiss();
  *memObj = retMemObj.release();
  return PI_SUCCESS;
}

pi_result hip_piMemGetInfo(pi_mem memObj, pi_mem_info queriedInfo,
                           size_t expectedQuerySize, void *queryOutput,
                           size_t *writtenQuerySize) {
  (void)memObj;
  (void)queriedInfo;
  (void)expectedQuerySize;
  (void)queryOutput;
  (void)writtenQuerySize;

  sycl::detail::pi::die("hip_piMemGetInfo not implemented");
}

/// Gets the native HIP handle of a PI mem object
///
/// \param[in] mem The PI mem to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI mem object.
///
/// \return PI_SUCCESS
pi_result hip_piextMemGetNativeHandle(pi_mem mem,
                                      pi_native_handle *nativeHandle) {
#if defined(__HIP_PLATFORM_NVIDIA__)
  if (sizeof(_pi_mem::mem_::buffer_mem_::native_type) >
      sizeof(pi_native_handle)) {
    // Check that all the upper bits that cannot be represented by
    // pi_native_handle are empty.
    // NOTE: The following shift might trigger a warning, but the check in the
    // if above makes sure that this does not underflow.
    _pi_mem::mem_::buffer_mem_::native_type upperBits =
        mem->mem_.buffer_mem_.get() >> (sizeof(pi_native_handle) * CHAR_BIT);
    if (upperBits) {
      // Return an error if any of the remaining bits is non-zero.
      return PI_ERROR_INVALID_MEM_OBJECT;
    }
  }
  *nativeHandle = static_cast<pi_native_handle>(mem->mem_.buffer_mem_.get());
#elif defined(__HIP_PLATFORM_AMD__)
  *nativeHandle =
      reinterpret_cast<pi_native_handle>(mem->mem_.buffer_mem_.get());
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
  return PI_SUCCESS;
}

/// Created a PI mem object from a HIP mem handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI mem object from.
/// \param[in] context The PI context of the memory allocation.
/// \param[in] ownNativeHandle Indicates if we own the native memory handle or
/// it came from interop that asked to not transfer the ownership to SYCL RT.
/// \param[out] mem Set to the PI mem object created from native handle.
///
/// \return TBD
pi_result hip_piextMemCreateWithNativeHandle(pi_native_handle nativeHandle,
                                             pi_context context,
                                             bool ownNativeHandle,
                                             pi_mem *mem) {
  (void)nativeHandle;
  (void)context;
  (void)ownNativeHandle;
  (void)mem;

  sycl::detail::pi::die(
      "Creation of PI mem from native handle not implemented");
  return {};
}

/// Created a PI image mem object from a HIP image mem handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI mem object from.
/// \param[in] context The PI context of the memory allocation.
/// \param[in] ownNativeHandle Indicates if we own the native memory handle or
/// it came from interop that asked to not transfer the ownership to SYCL RT.
/// \param[in] ImageFormat The format of the image.
/// \param[in] ImageDesc The description information for the image.
/// \param[out] mem Set to the PI mem object created from native handle.
///
/// \return TBD
pi_result hip_piextMemImageCreateWithNativeHandle(
    pi_native_handle nativeHandle, pi_context context, bool ownNativeHandle,
    const pi_image_format *ImageFormat, const pi_image_desc *ImageDesc,
    pi_mem *mem) {
  (void)nativeHandle;
  (void)context;
  (void)ownNativeHandle;
  (void)ImageFormat;
  (void)ImageDesc;
  (void)mem;

  sycl::detail::pi::die(
      "Creation of PI mem from native image handle not implemented");
  return {};
}

/// Creates a `pi_queue` object on the HIP backend.
/// Valid properties
/// * __SYCL_PI_HIP_USE_DEFAULT_STREAM -> hipStreamDefault
/// * __SYCL_PI_HIP_SYNC_WITH_DEFAULT -> hipStreamNonBlocking
/// \return Pi queue object mapping to a HIPStream
///
pi_result hip_piQueueCreate(pi_context context, pi_device device,
                            pi_queue_properties properties, pi_queue *queue) {
  try {
    std::unique_ptr<_pi_queue> queueImpl{nullptr};

    if (context->get_device() != device) {
      *queue = nullptr;
      return PI_ERROR_INVALID_DEVICE;
    }

    unsigned int flags = 0;

    const bool is_out_of_order =
        properties & PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    std::vector<hipStream_t> computeHipStreams(
        is_out_of_order ? _pi_queue::default_num_compute_streams : 1);
    std::vector<hipStream_t> transferHipStreams(
        is_out_of_order ? _pi_queue::default_num_transfer_streams : 0);

    queueImpl = std::unique_ptr<_pi_queue>(new _pi_queue{
        std::move(computeHipStreams), std::move(transferHipStreams), context,
        device, properties, flags});

    *queue = queueImpl.release();

    return PI_SUCCESS;
  } catch (pi_result err) {

    return err;

  } catch (...) {

    return PI_ERROR_OUT_OF_RESOURCES;
  }
}
pi_result hip_piextQueueCreate(pi_context Context, pi_device Device,
                               pi_queue_properties *Properties,
                               pi_queue *Queue) {
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
  return hip_piQueueCreate(Context, Device, Flags, Queue);
}

pi_result hip_piQueueGetInfo(pi_queue command_queue, pi_queue_info param_name,
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
  case PI_EXT_ONEAPI_QUEUE_INFO_EMPTY: {
    bool IsReady = command_queue->all_of([](hipStream_t s) -> bool {
      const hipError_t ret = hipStreamQuery(s);
      if (ret == hipSuccess)
        return true;

      if (ret == hipErrorNotReady)
        return false;

      PI_CHECK_ERROR(ret);
      return false;
    });
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   IsReady);
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Queue info request not implemented");
  return {};
}

pi_result hip_piQueueRetain(pi_queue command_queue) {
  assert(command_queue != nullptr);
  assert(command_queue->get_reference_count() > 0);

  command_queue->increment_reference_count();
  return PI_SUCCESS;
}

pi_result hip_piQueueRelease(pi_queue command_queue) {
  assert(command_queue != nullptr);

  if (command_queue->decrement_reference_count() > 0) {
    return PI_SUCCESS;
  }

  try {
    std::unique_ptr<_pi_queue> queueImpl(command_queue);

    ScopedContext active(command_queue->get_context());

    command_queue->for_each_stream([](hipStream_t s) {
      PI_CHECK_ERROR(hipStreamSynchronize(s));
      PI_CHECK_ERROR(hipStreamDestroy(s));
    });

    return PI_SUCCESS;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_OUT_OF_RESOURCES;
  }
}

pi_result hip_piQueueFinish(pi_queue command_queue) {

  // set default result to a negative result (avoid false-positve tests)
  pi_result result = PI_ERROR_OUT_OF_HOST_MEMORY;

  try {

    assert(command_queue !=
           nullptr); // need PI_ERROR_INVALID_EXTERNAL_HANDLE error code
    ScopedContext active(command_queue->get_context());

    command_queue->sync_streams<true>([&result](hipStream_t s) {
      result = PI_CHECK_ERROR(hipStreamSynchronize(s));
    });

  } catch (pi_result err) {

    result = err;

  } catch (...) {

    result = PI_ERROR_OUT_OF_RESOURCES;
  }

  return result;
}

// There is no HIP counterpart for queue flushing and we don't run into the
// same problem of having to flush cross-queue dependencies as some of the
// other plugins, so it can be left as no-op.
pi_result hip_piQueueFlush(pi_queue command_queue) {
  (void)command_queue;
  return PI_SUCCESS;
}

/// Gets the native HIP handle of a PI queue object
///
/// \param[in] queue The PI queue to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI queue object.
///
/// \return PI_SUCCESS
pi_result hip_piextQueueGetNativeHandle(pi_queue queue,
                                        pi_native_handle *nativeHandle,
                                        int32_t *NativeHandleDesc) {
  *NativeHandleDesc = 0;
  ScopedContext active(queue->get_context());
  *nativeHandle =
      reinterpret_cast<pi_native_handle>(queue->get_next_compute_stream());
  return PI_SUCCESS;
}

/// Created a PI queue object from a HIP queue handle.
/// TODO: Implement this.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create PI queue object from.
/// \param[in] context is the PI context of the queue.
/// \param[out] queue Set to the PI queue object created from native handle.
/// \param ownNativeHandle tells if SYCL RT should assume the ownership of
///        the native handle, if it can.
///
///
/// \return TBD
pi_result hip_piextQueueCreateWithNativeHandle(
    pi_native_handle nativeHandle, int32_t NativeHandleDesc, pi_context context,
    pi_device device, bool ownNativeHandle, pi_queue_properties *Properties,
    pi_queue *queue) {
  (void)nativeHandle;
  (void)NativeHandleDesc;
  (void)context;
  (void)device;
  (void)ownNativeHandle;
  (void)Properties;
  (void)queue;
  sycl::detail::pi::die(
      "Creation of PI queue from native handle not implemented");
  return {};
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

pi_result hip_piEventsWait(pi_uint32 num_events, const pi_event *event_list) {

  try {
    assert(num_events != 0);
    assert(event_list);
    if (num_events == 0) {
      return PI_ERROR_INVALID_VALUE;
    }

    if (!event_list) {
      return PI_ERROR_INVALID_EVENT;
    }

    auto context = event_list[0]->get_context();
    ScopedContext active(context);

    auto waitFunc = [context](pi_event event) -> pi_result {
      if (!event) {
        return PI_ERROR_INVALID_EVENT;
      }

      if (event->get_context() != context) {
        return PI_ERROR_INVALID_CONTEXT;
      }

      return event->wait();
    };
    return forLatestEvents(event_list, num_events, waitFunc);
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_ERROR_OUT_OF_RESOURCES;
  }
}

pi_result hip_piKernelCreate(pi_program program, const char *kernel_name,
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
    retErr = PI_ERROR_OUT_OF_HOST_MEMORY;
  }

  *kernel = retKernel.release();
  return retErr;
}

pi_result hip_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
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
        command_queue->device_, PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
        sizeof(maxThreadsPerBlock), maxThreadsPerBlock, nullptr);
    assert(retError == PI_SUCCESS);
    (void)retError;

    retError = pi2ur::piDeviceGetInfo(
        command_queue->device_, PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
        sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
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
        num_events_in_wait_list, event_wait_list, guard, &stream_token);
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
      retError = retImplEv->record();
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

/// \TODO Not implemented

pi_result hip_piMemImageCreate(pi_context context, pi_mem_flags flags,
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
    sycl::detail::pi::die(
        "hip_piMemImageCreate only supports RGBA channel order");
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
    sycl::detail::pi::die(
        "hip_piMemImageCreate given unsupported image_channel_data_type");
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
      return PI_ERROR_OUT_OF_HOST_MEMORY;
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
pi_result hip_piMemImageGetInfo(pi_mem image, pi_image_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  (void)image;
  (void)param_name;
  (void)param_value_size;
  (void)param_value;
  (void)param_value_size_ret;

  sycl::detail::pi::die("hip_piMemImageGetInfo not implemented");
  return {};
}

pi_result hip_piMemRetain(pi_mem mem) {
  assert(mem != nullptr);
  assert(mem->get_reference_count() > 0);
  mem->increment_reference_count();
  return PI_SUCCESS;
}

/// Not used as HIP backend only creates programs from binary.
/// See \ref hip_piclProgramCreateWithBinary.
///
pi_result hip_piclProgramCreateWithSource(pi_context context, pi_uint32 count,
                                          const char **strings,
                                          const size_t *lengths,
                                          pi_program *program) {
  (void)context;
  (void)count;
  (void)strings;
  (void)lengths;
  (void)program;

  sycl::detail::pi::hipPrint("hip_piclProgramCreateWithSource not implemented");
  return PI_ERROR_INVALID_OPERATION;
}

/// Loads the images from a PI program into a HIPmodule that can be
/// used later on to extract functions (kernels).
/// See \ref _pi_program for implementation details.
///
pi_result hip_piProgramBuild(
    pi_program program, [[maybe_unused]] pi_uint32 num_devices,
    [[maybe_unused]] const pi_device *device_list, const char *options,
    [[maybe_unused]] void (*pfn_notify)(pi_program program, void *user_data),
    [[maybe_unused]] void *user_data) {

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
pi_result hip_piProgramCreate(pi_context context, const void *il, size_t length,
                              pi_program *res_program) {
  (void)context;
  (void)il;
  (void)length;
  (void)res_program;

  sycl::detail::pi::die("hip_piProgramCreate not implemented");
  return {};
}

/// Loads images from a list of PTX or HIPBIN binaries.
/// Note: No calls to HIP driver API in this function, only store binaries
/// for later.
///
/// Note: Only supports one device
///
pi_result hip_piProgramCreateWithBinary(
    pi_context context, [[maybe_unused]] pi_uint32 num_devices,
    [[maybe_unused]] const pi_device *device_list, const size_t *lengths,
    const unsigned char **binaries, size_t num_metadata_entries,
    const pi_device_binary_property *metadata, pi_int32 *binary_status,
    pi_program *program) {
  (void)num_metadata_entries;
  (void)metadata;
  (void)binary_status;

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

  // TODO: Set metadata here and use reqd_work_group_size information.
  // See cuda_piProgramCreateWithBinary

  const bool has_length = (lengths != nullptr);
  size_t length = has_length
                      ? lengths[0]
                      : strlen(reinterpret_cast<const char *>(binaries[0])) + 1;

  assert(length != 0);

  retProgram->set_binary(reinterpret_cast<const char *>(binaries[0]), length);

  *program = retProgram.release();

  return retError;
}

pi_result hip_piProgramGetInfo(pi_program program, pi_program_info param_name,
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
  sycl::detail::pi::die("Program info request not implemented");
  return {};
}

pi_result hip_piProgramLink(pi_context context, pi_uint32 num_devices,
                            const pi_device *device_list, const char *options,
                            pi_uint32 num_input_programs,
                            const pi_program *input_programs,
                            void (*pfn_notify)(pi_program program,
                                               void *user_data),
                            void *user_data, pi_program *ret_program) {
  (void)context;
  (void)num_devices;
  (void)device_list;
  (void)options;
  (void)num_input_programs;
  (void)input_programs;
  (void)pfn_notify;
  (void)user_data;
  (void)ret_program;
  sycl::detail::pi::die(
      "hip_piProgramLink: linking not supported with hip backend");
  return {};
}

/// Creates a new program that is the outcome of the compilation of the headers
///  and the program.
/// \TODO Implement asynchronous compilation
///
pi_result hip_piProgramCompile(
    pi_program program, [[maybe_unused]] pi_uint32 num_devices,
    [[maybe_unused]] const pi_device *device_list, const char *options,
    [[maybe_unused]] pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    [[maybe_unused]] void (*pfn_notify)(pi_program program, void *user_data),
    [[maybe_unused]] void *user_data) {
  (void)input_headers;
  (void)header_include_names;

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

pi_result hip_piProgramGetBuildInfo(pi_program program, pi_device device,
                                    pi_program_build_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {
  (void)device;

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
  sycl::detail::pi::die("Program Build info request not implemented");
  return {};
}

pi_result hip_piProgramRetain(pi_program program) {
  assert(program != nullptr);
  assert(program->get_reference_count() > 0);
  program->increment_reference_count();
  return PI_SUCCESS;
}

/// Decreases the reference count of a pi_program object.
/// When the reference count reaches 0, it unloads the module from
/// the context.
pi_result hip_piProgramRelease(pi_program program) {
  assert(program != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  assert(program->get_reference_count() != 0 &&
         "Reference count overflow detected in hip_piProgramRelease.");

  // decrement ref count. If it is 0, delete the program.
  if (program->decrement_reference_count() == 0) {

    std::unique_ptr<_pi_program> program_ptr{program};

    pi_result result = PI_ERROR_INVALID_PROGRAM;

    try {
      ScopedContext active(program->get_context());
      auto hipModule = program->get();
      result = PI_CHECK_ERROR(hipModuleUnload(hipModule));
    } catch (...) {
      result = PI_ERROR_OUT_OF_RESOURCES;
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
pi_result hip_piextProgramGetNativeHandle(pi_program program,
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
/// \param[in] ownNativeHandle tells if should assume the ownership of
///            the native handle.
/// \param[out] program Set to the PI program object created from native handle.
///
/// \return TBD
pi_result hip_piextProgramCreateWithNativeHandle(pi_native_handle nativeHandle,
                                                 pi_context context,
                                                 bool ownNativeHandle,
                                                 pi_program *program) {
  (void)nativeHandle;
  (void)context;
  (void)ownNativeHandle;
  (void)program;

  sycl::detail::pi::die(
      "Creation of PI program from native handle not implemented");
  return {};
}

pi_result hip_piKernelGetInfo(pi_kernel kernel, pi_kernel_info param_name,
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

  return PI_ERROR_INVALID_KERNEL;
}

pi_result hip_piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                   pi_kernel_group_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {

  // here we want to query about a kernel's hip blocks!

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
      size_t global_work_size[3] = {0, 0, 0};

      int max_block_dimX{0}, max_block_dimY{0}, max_block_dimZ{0};
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&max_block_dimX, hipDeviceAttributeMaxBlockDimX,
                                device->get()) == hipSuccess);
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&max_block_dimY, hipDeviceAttributeMaxBlockDimY,
                                device->get()) == hipSuccess);
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&max_block_dimZ, hipDeviceAttributeMaxBlockDimZ,
                                device->get()) == hipSuccess);

      int max_grid_dimX{0}, max_grid_dimY{0}, max_grid_dimZ{0};
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&max_grid_dimX, hipDeviceAttributeMaxGridDimX,
                                device->get()) == hipSuccess);
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&max_grid_dimY, hipDeviceAttributeMaxGridDimY,
                                device->get()) == hipSuccess);
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&max_grid_dimZ, hipDeviceAttributeMaxGridDimZ,
                                device->get()) == hipSuccess);

      global_work_size[0] = max_block_dimX * max_grid_dimX;
      global_work_size[1] = max_block_dimY * max_grid_dimY;
      global_work_size[2] = max_block_dimZ * max_grid_dimZ;
      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, global_work_size);
    }
    case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
      int max_threads = 0;
      sycl::detail::pi::assertion(
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
      sycl::detail::pi::assertion(
          hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                              kernel->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
      // Work groups should be multiples of the warp size
      int warpSize = 0;
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                                device->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<size_t>(warpSize));
    }
    case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
      // OpenCL PRIVATE == HIP LOCAL
      int bytes = 0;
      sycl::detail::pi::assertion(
          hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                              kernel->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));
    }
    case PI_KERNEL_GROUP_INFO_NUM_REGS: {
      sycl::detail::pi::die("PI_KERNEL_GROUP_INFO_NUM_REGS in "
                            "piKernelGetGroupInfo not implemented\n");
      return {};
    }

    default:
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }

  return PI_ERROR_INVALID_KERNEL;
}

pi_result hip_piKernelGetSubGroupInfo(
    pi_kernel kernel, pi_device device, pi_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  (void)input_value_size;
  (void)input_value;

  if (kernel != nullptr) {
    switch (param_name) {
    case PI_KERNEL_MAX_SUB_GROUP_SIZE: {
      // Sub-group size is equivalent to warp size
      int warpSize = 0;
      sycl::detail::pi::assertion(
          hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                                device->get()) == hipSuccess);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<uint32_t>(warpSize));
    }
    case PI_KERNEL_MAX_NUM_SUB_GROUPS: {
      // Number of sub-groups = max block size / warp size + possible remainder
      int max_threads = 0;
      sycl::detail::pi::assertion(
          hipFuncGetAttribute(&max_threads,
                              HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                              kernel->get()) == hipSuccess);
      int warpSize = 0;
      hip_piKernelGetSubGroupInfo(kernel, device, PI_KERNEL_MAX_SUB_GROUP_SIZE,
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
  return PI_ERROR_INVALID_KERNEL;
}

pi_result hip_piKernelRetain(pi_kernel kernel) {
  assert(kernel != nullptr);
  assert(kernel->get_reference_count() > 0u);

  kernel->increment_reference_count();
  return PI_SUCCESS;
}

pi_result hip_piKernelRelease(pi_kernel kernel) {
  assert(kernel != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  assert(kernel->get_reference_count() != 0 &&
         "Reference count overflow detected in hip_piKernelRelease.");

  // decrement ref count. If it is 0, delete the program.
  if (kernel->decrement_reference_count() == 0) {
    // no internal hip resources to clean up. Just delete it.
    delete kernel;
    return PI_SUCCESS;
  }

  return PI_SUCCESS;
}

// A NOP for the HIP backend
pi_result hip_piKernelSetExecInfo(pi_kernel kernel,
                                  pi_kernel_exec_info param_name,
                                  size_t param_value_size,
                                  const void *param_value) {
  (void)kernel;
  (void)param_name;
  (void)param_value_size;
  (void)param_value;

  return PI_SUCCESS;
}

pi_result hip_piextProgramSetSpecializationConstant(pi_program, pi_uint32,
                                                    size_t, const void *) {
  // This entry point is only used for native specialization constants (SPIR-V),
  // and the HIP plugin is AOT only so this entry point is not supported.
  sycl::detail::pi::die("Native specialization constants are not supported");
  return {};
}

pi_result hip_piextKernelSetArgPointer(pi_kernel kernel, pi_uint32 arg_index,
                                       size_t arg_size, const void *arg_value) {
  kernel->set_kernel_arg(arg_index, arg_size, arg_value);
  return PI_SUCCESS;
}

//
// Events
//
pi_result hip_piEventCreate(pi_context context, pi_event *event) {
  (void)context;
  (void)event;

  sycl::detail::pi::die("PI Event Create not implemented in HIP backend");
}

pi_result hip_piEventGetInfo(pi_event event, pi_event_info param_name,
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

  return PI_ERROR_INVALID_EVENT;
}

/// Obtain profiling information from PI HIP events
/// Timings from HIP are only elapsed time.
pi_result hip_piEventGetProfilingInfo(pi_event event,
                                      pi_profiling_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {

  assert(event != nullptr);

  pi_queue queue = event->get_queue();
  if (queue == nullptr ||
      !(queue->properties_ & PI_QUEUE_FLAG_PROFILING_ENABLE)) {
    return PI_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  switch (param_name) {
  case PI_PROFILING_INFO_COMMAND_QUEUED:
  case PI_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No user for this case
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
  sycl::detail::pi::die("Event Profiling info request not implemented");
  return {};
}

pi_result hip_piEventSetCallback(pi_event event,
                                 pi_int32 command_exec_callback_type,
                                 pfn_notify notify, void *user_data) {
  (void)event;
  (void)command_exec_callback_type;
  (void)notify;
  (void)user_data;

  sycl::detail::pi::die("Event Callback not implemented in HIP backend");
  return PI_SUCCESS;
}

pi_result hip_piEventSetStatus(pi_event event, pi_int32 execution_status) {
  (void)event;
  (void)execution_status;

  sycl::detail::pi::die("Event Set Status not implemented in HIP backend");
  return PI_ERROR_INVALID_VALUE;
}

pi_result hip_piEventRetain(pi_event event) {
  assert(event != nullptr);

  const auto refCount = event->increment_reference_count();

  sycl::detail::pi::assertion(
      refCount != 0, "Reference count overflow detected in hip_piEventRetain.");

  return PI_SUCCESS;
}

pi_result hip_piEventRelease(pi_event event) {
  assert(event != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  sycl::detail::pi::assertion(
      event->get_reference_count() != 0,
      "Reference count overflow detected in hip_piEventRelease.");

  // decrement ref count. If it is 0, delete the event.
  if (event->decrement_reference_count() == 0) {
    std::unique_ptr<_pi_event> event_ptr{event};
    pi_result result = PI_ERROR_INVALID_EVENT;
    try {
      ScopedContext active(event->get_context());
      result = event->release();
    } catch (...) {
      result = PI_ERROR_OUT_OF_RESOURCES;
    }
    return result;
  }

  return PI_SUCCESS;
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
        num_events_in_wait_list, event_wait_list, guard, &stream_token);
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
        forLatestEvents(event_wait_list, num_events_in_wait_list,
                        [hipStream](pi_event event) -> pi_result {
                          if (event->get_queue()->has_been_synchronized(
                                  event->get_compute_stream_token())) {
                            return PI_SUCCESS;
                          } else {
                            return PI_CHECK_ERROR(
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

/// Gets the native HIP handle of a PI event object
///
/// \param[in] event The PI event to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the PI event object.
///
/// \return PI_SUCCESS on success. PI_ERROR_INVALID_EVENT if given a user event.
pi_result hip_piextEventGetNativeHandle(pi_event event,
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
pi_result hip_piextEventCreateWithNativeHandle(pi_native_handle nativeHandle,
                                               pi_context context,
                                               bool ownNativeHandle,
                                               pi_event *event) {
  (void)nativeHandle;
  (void)context;
  (void)ownNativeHandle;
  (void)event;

  sycl::detail::pi::die(
      "Creation of PI event from native handle not implemented");
  return {};
}

/// Creates a PI sampler object
///
/// \param[in] context The context the sampler is created for.
/// \param[in] sampler_properties The properties for the sampler.
/// \param[out] result_sampler Set to the resulting sampler object.
///
/// \return PI_SUCCESS on success. PI_ERROR_INVALID_VALUE if given an invalid
/// property
///         or if there is multiple of properties from the same category.
pi_result hip_piSamplerCreate(pi_context context,
                              const pi_sampler_properties *sampler_properties,
                              pi_sampler *result_sampler) {
  std::unique_ptr<_pi_sampler> retImplSampl{new _pi_sampler(context)};

  bool propSeen[3] = {false, false, false};
  for (size_t i = 0; sampler_properties[i] != 0; i += 2) {
    switch (sampler_properties[i]) {
    case PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS:
      if (propSeen[0]) {
        return PI_ERROR_INVALID_VALUE;
      }
      propSeen[0] = true;
      retImplSampl->props_ |= sampler_properties[i + 1];
      break;
    case PI_SAMPLER_PROPERTIES_FILTER_MODE:
      if (propSeen[1]) {
        return PI_ERROR_INVALID_VALUE;
      }
      propSeen[1] = true;
      retImplSampl->props_ |=
          (sampler_properties[i + 1] - PI_SAMPLER_FILTER_MODE_NEAREST) << 1;
      break;
    case PI_SAMPLER_PROPERTIES_ADDRESSING_MODE:
      if (propSeen[2]) {
        return PI_ERROR_INVALID_VALUE;
      }
      propSeen[2] = true;
      retImplSampl->props_ |=
          (sampler_properties[i + 1] - PI_SAMPLER_ADDRESSING_MODE_NONE) << 2;
      break;
    default:
      return PI_ERROR_INVALID_VALUE;
    }
  }

  if (!propSeen[0]) {
    retImplSampl->props_ |= PI_TRUE;
  }
  // Default filter mode to CL_FILTER_NEAREST
  if (!propSeen[2]) {
    retImplSampl->props_ |=
        (PI_SAMPLER_ADDRESSING_MODE_CLAMP % PI_SAMPLER_ADDRESSING_MODE_NONE)
        << 2;
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
pi_result hip_piSamplerGetInfo(pi_sampler sampler, pi_sampler_info param_name,
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
pi_result hip_piSamplerRetain(pi_sampler sampler) {
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
pi_result hip_piSamplerRelease(pi_sampler sampler) {
  assert(sampler != nullptr);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  sycl::detail::pi::assertion(
      sampler->get_reference_count() != 0,
      "Reference count overflow detected in hip_piSamplerRelease.");

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
      result = retImplEv->start();
    }

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
      result = retImplEv->start();
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

/// USM: Implements USM Host allocations using HIP Pinned Memory
///
pi_result
hip_piextUSMHostAlloc(void **result_ptr, pi_context context,
                      [[maybe_unused]] pi_usm_mem_properties *properties,
                      size_t size, [[maybe_unused]] pi_uint32 alignment) {
  assert(result_ptr != nullptr);
  assert(context != nullptr);
  assert(properties == nullptr || *properties == 0);
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
pi_result
hip_piextUSMDeviceAlloc(void **result_ptr, pi_context context,
                        [[maybe_unused]] pi_device device,
                        [[maybe_unused]] pi_usm_mem_properties *properties,
                        size_t size, [[maybe_unused]] pi_uint32 alignment) {
  assert(result_ptr != nullptr);
  assert(context != nullptr);
  assert(device != nullptr);
  assert(properties == nullptr || *properties == 0);
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
pi_result
hip_piextUSMSharedAlloc(void **result_ptr, pi_context context,
                        [[maybe_unused]] pi_device device,
                        [[maybe_unused]] pi_usm_mem_properties *properties,
                        size_t size, [[maybe_unused]] pi_uint32 alignment) {
  assert(result_ptr != nullptr);
  assert(context != nullptr);
  assert(device != nullptr);
  assert(properties == nullptr || *properties == 0);
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
pi_result hip_piextUSMFree(pi_context context, void *ptr) {

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
        num_events_in_waitlist, events_waitlist, guard, &stream_token);
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
      result = event_ptr->record();
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
      result = event_ptr->record();
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
pi_result hip_piextUSMGetMemAllocInfo(pi_context context, const void *ptr,
                                      pi_mem_alloc_info param_name,
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
      return PI_ERROR_INVALID_VALUE;
    }
    case PI_MEM_ALLOC_SIZE: {
      return PI_ERROR_INVALID_VALUE;
    }

    case PI_MEM_ALLOC_DEVICE: {
      // get device index associated with this pointer
      result = PI_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, ptr));
      int device_idx = hipPointerAttributeType.device;

      // currently each device is in its own platform, so find the platform at
      // the same index
      std::vector<pi_platform> platforms;
      platforms.resize(device_idx + 1);
      result = pi2ur::piPlatformsGet(device_idx + 1, platforms.data(), nullptr);

      // get the device from the platform
      pi_device device =
          reinterpret_cast<pi_device>(platforms[device_idx]->devices_[0].get());
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     device);
    }
    }
  } catch (pi_result error) {
    result = error;
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
  _PI_CL(piQueueCreate, hip_piQueueCreate)
  _PI_CL(piextQueueCreate, hip_piextQueueCreate)
  _PI_CL(piQueueGetInfo, hip_piQueueGetInfo)
  _PI_CL(piQueueFinish, hip_piQueueFinish)
  _PI_CL(piQueueFlush, hip_piQueueFlush)
  _PI_CL(piQueueRetain, hip_piQueueRetain)
  _PI_CL(piQueueRelease, hip_piQueueRelease)
  _PI_CL(piextQueueGetNativeHandle, hip_piextQueueGetNativeHandle)
  _PI_CL(piextQueueCreateWithNativeHandle, hip_piextQueueCreateWithNativeHandle)
  // Memory
  _PI_CL(piMemBufferCreate, hip_piMemBufferCreate)
  _PI_CL(piMemImageCreate, hip_piMemImageCreate)
  _PI_CL(piMemGetInfo, hip_piMemGetInfo)
  _PI_CL(piMemImageGetInfo, hip_piMemImageGetInfo)
  _PI_CL(piMemRetain, hip_piMemRetain)
  _PI_CL(piMemRelease, hip_piMemRelease)
  _PI_CL(piMemBufferPartition, hip_piMemBufferPartition)
  _PI_CL(piextMemGetNativeHandle, hip_piextMemGetNativeHandle)
  _PI_CL(piextMemCreateWithNativeHandle, hip_piextMemCreateWithNativeHandle)
  // Program
  _PI_CL(piProgramCreate, hip_piProgramCreate)
  _PI_CL(piclProgramCreateWithSource, hip_piclProgramCreateWithSource)
  _PI_CL(piProgramCreateWithBinary, hip_piProgramCreateWithBinary)
  _PI_CL(piProgramGetInfo, hip_piProgramGetInfo)
  _PI_CL(piProgramCompile, hip_piProgramCompile)
  _PI_CL(piProgramBuild, hip_piProgramBuild)
  _PI_CL(piProgramLink, hip_piProgramLink)
  _PI_CL(piProgramGetBuildInfo, hip_piProgramGetBuildInfo)
  _PI_CL(piProgramRetain, hip_piProgramRetain)
  _PI_CL(piProgramRelease, hip_piProgramRelease)
  _PI_CL(piextProgramGetNativeHandle, hip_piextProgramGetNativeHandle)
  _PI_CL(piextProgramCreateWithNativeHandle,
         hip_piextProgramCreateWithNativeHandle)
  // Kernel
  _PI_CL(piKernelCreate, hip_piKernelCreate)
  _PI_CL(piKernelSetArg, hip_piKernelSetArg)
  _PI_CL(piKernelGetInfo, hip_piKernelGetInfo)
  _PI_CL(piKernelGetGroupInfo, hip_piKernelGetGroupInfo)
  _PI_CL(piKernelGetSubGroupInfo, hip_piKernelGetSubGroupInfo)
  _PI_CL(piKernelRetain, hip_piKernelRetain)
  _PI_CL(piKernelRelease, hip_piKernelRelease)
  _PI_CL(piKernelSetExecInfo, hip_piKernelSetExecInfo)
  _PI_CL(piextProgramSetSpecializationConstant,
         hip_piextProgramSetSpecializationConstant)
  _PI_CL(piextKernelSetArgPointer, hip_piextKernelSetArgPointer)
  // Event
  _PI_CL(piEventCreate, hip_piEventCreate)
  _PI_CL(piEventGetInfo, hip_piEventGetInfo)
  _PI_CL(piEventGetProfilingInfo, hip_piEventGetProfilingInfo)
  _PI_CL(piEventsWait, hip_piEventsWait)
  _PI_CL(piEventSetCallback, hip_piEventSetCallback)
  _PI_CL(piEventSetStatus, hip_piEventSetStatus)
  _PI_CL(piEventRetain, hip_piEventRetain)
  _PI_CL(piEventRelease, hip_piEventRelease)
  _PI_CL(piextEventGetNativeHandle, hip_piextEventGetNativeHandle)
  _PI_CL(piextEventCreateWithNativeHandle, hip_piextEventCreateWithNativeHandle)
  // Sampler
  _PI_CL(piSamplerCreate, hip_piSamplerCreate)
  _PI_CL(piSamplerGetInfo, hip_piSamplerGetInfo)
  _PI_CL(piSamplerRetain, hip_piSamplerRetain)
  _PI_CL(piSamplerRelease, hip_piSamplerRelease)
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
  _PI_CL(piextUSMHostAlloc, hip_piextUSMHostAlloc)
  _PI_CL(piextUSMDeviceAlloc, hip_piextUSMDeviceAlloc)
  _PI_CL(piextUSMSharedAlloc, hip_piextUSMSharedAlloc)
  _PI_CL(piextUSMFree, hip_piextUSMFree)
  _PI_CL(piextUSMEnqueueMemset, hip_piextUSMEnqueueMemset)
  _PI_CL(piextUSMEnqueueMemcpy, hip_piextUSMEnqueueMemcpy)
  _PI_CL(piextUSMEnqueuePrefetch, hip_piextUSMEnqueuePrefetch)
  _PI_CL(piextUSMEnqueueMemAdvise, hip_piextUSMEnqueueMemAdvise)
  _PI_CL(piextUSMEnqueueMemcpy2D, hip_piextUSMEnqueueMemcpy2D)
  _PI_CL(piextUSMEnqueueFill2D, hip_piextUSMEnqueueFill2D)
  _PI_CL(piextUSMEnqueueMemset2D, hip_piextUSMEnqueueMemset2D)
  _PI_CL(piextUSMGetMemAllocInfo, hip_piextUSMGetMemAllocInfo)
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

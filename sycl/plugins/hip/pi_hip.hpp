//===-- pi_hip.hpp - HIP Plugin -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_hip HIP Plugin
/// \ingroup sycl_pi

/// \file pi_hip.hpp
/// Declarations for HIP Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying HIP runtime.
///
/// \ingroup sycl_pi_hip

#ifndef PI_HIP_HPP
#define PI_HIP_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_HIP_PLUGIN_VERSION 1

#define _PI_HIP_PLUGIN_VERSION_STRING                                          \
  _PI_PLUGIN_VERSION_STRING(_PI_HIP_PLUGIN_VERSION)

#include "sycl/detail/pi.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <hip/hip_runtime.h>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>

#include <ur/adapters/hip/context.hpp>
#include <ur/adapters/hip/device.hpp>
#include <ur/adapters/hip/memory.hpp>
#include <ur/adapters/hip/platform.hpp>

#include "pi2ur.hpp"

extern "C" {

/// \cond INGORE_BLOCK_IN_DOXYGEN
pi_result hip_piProgramRetain(pi_program);
pi_result hip_piProgramRelease(pi_program);
pi_result hip_piQueueRelease(pi_queue);
pi_result hip_piQueueRetain(pi_queue);
pi_result hip_piKernelRetain(pi_kernel);
pi_result hip_piKernelRelease(pi_kernel);
/// \endcond
}

using _pi_stream_guard = std::unique_lock<std::mutex>;

/// A PI platform stores all known PI devices,
///  in the HIP plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct _pi_platform : ur_platform_handle_t_ {
  using ur_platform_handle_t_::ur_platform_handle_t_;
};

/// PI device mapping to a hipDevice_t.
/// Includes an observer pointer to the platform,
/// and implements the reference counting semantics since
/// HIP objects are not refcounted.
///
struct _pi_device : ur_device_handle_t_ {
  using ur_device_handle_t_::ur_device_handle_t_;
};

/// PI context mapping to a HIP context object.
///
/// There is no direct mapping between a HIP context and a PI context,
/// main differences described below:
///
/// <b> HIP context vs PI context </b>
///
/// One of the main differences between the PI API and the HIP driver API is
/// that the second modifies the state of the threads by assigning
/// `hipCtx_t` objects to threads. `hipCtx_t` objects store data associated
/// with a given device and control access to said device from the user side.
/// PI API context are objects that are passed to functions, and not bound
/// to threads.
/// The _pi_context object doesn't implement this behavior, only holds the
/// HIP context data. The RAII object \ref ScopedContext implements the active
/// context behavior.
///
/// <b> Primary vs User-defined context </b>
///
/// HIP has two different types of context, the Primary context,
/// which is usable by all threads on a given process for a given device, and
/// the aforementioned custom contexts.
/// HIP documentation, and performance analysis, indicates it is recommended
/// to use Primary context whenever possible.
/// Primary context is used as well by the HIP Runtime API.
/// For PI applications to interop with HIP Runtime API, they have to use
/// the primary context - and make that active in the thread.
/// The `_pi_context` object can be constructed with a `kind` parameter
/// that allows to construct a Primary or `user-defined` context, so that
/// the PI object interface is always the same.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the PI Context can store a number of callback functions that will be
///  called upon destruction of the PI Context.
///  See proposal for details.
///
struct _pi_context : ur_context_handle_t_ {
  using ur_context_handle_t_::ur_context_handle_t_;
};

/// PI Mem mapping to HIP memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the HIP backend.
/// Keeps tracks of all mapped regions used for Map/Unmap calls.
/// Only one region can be active at the same time per allocation.
struct _pi_mem : ur_mem_handle_t_ {
  using ur_mem_handle_t_::ur_mem_handle_t_;
};

/// PI queue mapping on to hipStream_t objects.
///
struct _pi_queue {
  using native_type = hipStream_t;
  static constexpr int default_num_compute_streams = 64;
  static constexpr int default_num_transfer_streams = 16;

  std::vector<native_type> compute_streams_;
  std::vector<native_type> transfer_streams_;
  // delay_compute_ keeps track of which streams have been recently reused and
  // their next use should be delayed. If a stream has been recently reused it
  // will be skipped the next time it would be selected round-robin style. When
  // skipped, its delay flag is cleared.
  std::vector<bool> delay_compute_;
  // keep track of which streams have applied barrier
  std::vector<bool> compute_applied_barrier_;
  std::vector<bool> transfer_applied_barrier_;
  _pi_context *context_;
  _pi_device *device_;
  pi_queue_properties properties_;
  hipEvent_t barrier_event_ = nullptr;
  hipEvent_t barrier_tmp_event_ = nullptr;
  std::atomic_uint32_t refCount_;
  std::atomic_uint32_t eventCount_;
  std::atomic_uint32_t compute_stream_idx_;
  std::atomic_uint32_t transfer_stream_idx_;
  unsigned int num_compute_streams_;
  unsigned int num_transfer_streams_;
  unsigned int last_sync_compute_streams_;
  unsigned int last_sync_transfer_streams_;
  unsigned int flags_;
  // When compute_stream_sync_mutex_ and compute_stream_mutex_ both need to be
  // locked at the same time, compute_stream_sync_mutex_ should be locked first
  // to avoid deadlocks
  std::mutex compute_stream_sync_mutex_;
  std::mutex compute_stream_mutex_;
  std::mutex transfer_stream_mutex_;
  std::mutex barrier_mutex_;

  _pi_queue(std::vector<native_type> &&compute_streams,
            std::vector<native_type> &&transfer_streams, _pi_context *context,
            _pi_device *device, pi_queue_properties properties,
            unsigned int flags)
      : compute_streams_{std::move(compute_streams)},
        transfer_streams_{std::move(transfer_streams)},
        delay_compute_(compute_streams_.size(), false),
        compute_applied_barrier_(compute_streams_.size()),
        transfer_applied_barrier_(transfer_streams_.size()), context_{context},
        device_{device}, properties_{properties}, refCount_{1}, eventCount_{0},
        compute_stream_idx_{0}, transfer_stream_idx_{0},
        num_compute_streams_{0}, num_transfer_streams_{0},
        last_sync_compute_streams_{0}, last_sync_transfer_streams_{0},
        flags_(flags) {
    pi2ur::piContextRetain(context_);
    pi2ur::piDeviceRetain(device_);
  }

  ~_pi_queue() {
    pi2ur::piContextRelease(context_);
    pi2ur::piDeviceRelease(device_);
  }

  void compute_stream_wait_for_barrier_if_needed(hipStream_t stream,
                                                 pi_uint32 stream_i);
  void transfer_stream_wait_for_barrier_if_needed(hipStream_t stream,
                                                  pi_uint32 stream_i);

  // get_next_compute/transfer_stream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type get_next_compute_stream(pi_uint32 *stream_token = nullptr);
  // this overload tries select a stream that was used by one of dependancies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type get_next_compute_stream(pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      _pi_stream_guard &guard,
                                      pi_uint32 *stream_token = nullptr);
  native_type get_next_transfer_stream();
  native_type get() { return get_next_compute_stream(); };

  bool has_been_synchronized(pi_uint32 stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<pi_uint32>::max()) {
      return false;
    }
    return last_sync_compute_streams_ > stream_token;
  }

  bool can_reuse_stream(pi_uint32 stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<pi_uint32>::max()) {
      return false;
    }
    // If the command represented by the stream token was not the last command
    // enqueued to the stream we can not reuse the stream - we need to allow for
    // commands enqueued after it and the one we are about to enqueue to run
    // concurrently
    bool is_last_command =
        (compute_stream_idx_ - stream_token) <= compute_streams_.size();
    // If there was a barrier enqueued to the queue after the command
    // represented by the stream token we should not reuse the stream, as we can
    // not take that stream into account for the bookkeeping for the next
    // barrier - such a stream would not be synchronized with. Performance-wise
    // it does not matter that we do not reuse the stream, as the work
    // represented by the stream token is guaranteed to be complete by the
    // barrier before any work we are about to enqueue to the stream will start,
    // so the event does not need to be synchronized with.
    return is_last_command && !has_been_synchronized(stream_token);
  }

  template <typename T> bool all_of(T &&f) {
    {
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(compute_streams_.size()),
                   num_compute_streams_);
      if (!std::all_of(compute_streams_.begin(), compute_streams_.begin() + end,
                       f))
        return false;
    }
    {
      std::lock_guard<std::mutex> transfer_guard(transfer_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(transfer_streams_.size()),
                   num_transfer_streams_);
      if (!std::all_of(transfer_streams_.begin(),
                       transfer_streams_.begin() + end, f))
        return false;
    }
    return true;
  }

  template <typename T> void for_each_stream(T &&f) {
    {
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(compute_streams_.size()),
                   num_compute_streams_);
      for (unsigned int i = 0; i < end; i++) {
        f(compute_streams_[i]);
      }
    }
    {
      std::lock_guard<std::mutex> transfer_guard(transfer_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(transfer_streams_.size()),
                   num_transfer_streams_);
      for (unsigned int i = 0; i < end; i++) {
        f(transfer_streams_[i]);
      }
    }
  }

  template <bool ResetUsed = false, typename T> void sync_streams(T &&f) {
    auto sync_compute = [&f, &streams = compute_streams_,
                         &delay = delay_compute_](unsigned int start,
                                                  unsigned int stop) {
      for (unsigned int i = start; i < stop; i++) {
        f(streams[i]);
        delay[i] = false;
      }
    };
    auto sync_transfer = [&f, &streams = transfer_streams_](unsigned int start,
                                                            unsigned int stop) {
      for (unsigned int i = start; i < stop; i++) {
        f(streams[i]);
      }
    };
    {
      unsigned int size = static_cast<unsigned int>(compute_streams_.size());
      std::lock_guard<std::mutex> compute_sync_guard(
          compute_stream_sync_mutex_);
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int start = last_sync_compute_streams_;
      unsigned int end = num_compute_streams_ < size
                             ? num_compute_streams_
                             : compute_stream_idx_.load();
      if (end - start >= size) {
        sync_compute(0, size);
      } else {
        start %= size;
        end %= size;
        if (start < end) {
          sync_compute(start, end);
        } else {
          sync_compute(start, size);
          sync_compute(0, end);
        }
      }
      if (ResetUsed) {
        last_sync_compute_streams_ = end;
      }
    }
    {
      unsigned int size = static_cast<unsigned int>(transfer_streams_.size());
      if (size > 0) {
        std::lock_guard<std::mutex> transfer_guard(transfer_stream_mutex_);
        unsigned int start = last_sync_transfer_streams_;
        unsigned int end = num_transfer_streams_ < size
                               ? num_transfer_streams_
                               : transfer_stream_idx_.load();
        if (end - start >= size) {
          sync_transfer(0, size);
        } else {
          start %= size;
          end %= size;
          if (start < end) {
            sync_transfer(start, end);
          } else {
            sync_transfer(start, size);
            sync_transfer(0, end);
          }
        }
        if (ResetUsed) {
          last_sync_transfer_streams_ = end;
        }
      }
    }
  }

  _pi_context *get_context() const { return context_; };

  _pi_device *get_device() const { return device_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  pi_uint32 get_next_event_id() noexcept { return ++eventCount_; }
};

typedef void (*pfn_notify)(pi_event event, pi_int32 eventCommandStatus,
                           void *userData);
/// PI Event mapping to hipEvent_t
///
struct _pi_event {
public:
  using native_type = hipEvent_t;

  pi_result record();

  pi_result wait();

  pi_result start();

  native_type get() const noexcept { return evEnd_; };

  pi_queue get_queue() const noexcept { return queue_; }

  hipStream_t get_stream() const noexcept { return stream_; }

  pi_uint32 get_compute_stream_token() const noexcept { return streamToken_; }

  pi_command_type get_command_type() const noexcept { return commandType_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  bool is_recorded() const noexcept { return isRecorded_; }

  bool is_started() const noexcept { return isStarted_; }

  bool is_completed() const noexcept;

  pi_int32 get_execution_status() const noexcept {

    if (!is_recorded()) {
      return PI_EVENT_SUBMITTED;
    }

    if (!is_completed()) {
      return PI_EVENT_RUNNING;
    }
    return PI_EVENT_COMPLETE;
  }

  pi_context get_context() const noexcept { return context_; };

  pi_uint32 increment_reference_count() { return ++refCount_; }

  pi_uint32 decrement_reference_count() { return --refCount_; }

  pi_uint32 get_event_id() const noexcept { return eventId_; }

  // Returns the counter time when the associated command(s) were enqueued
  //
  pi_uint64 get_queued_time() const;

  // Returns the counter time when the associated command(s) started execution
  //
  pi_uint64 get_start_time() const;

  // Returns the counter time when the associated command(s) completed
  //
  pi_uint64 get_end_time() const;

  // construct a native HIP. This maps closely to the underlying HIP event.
  static pi_event
  make_native(pi_command_type type, pi_queue queue, hipStream_t stream,
              pi_uint32 stream_token = std::numeric_limits<pi_uint32>::max()) {
    return new _pi_event(type, queue->get_context(), queue, stream,
                         stream_token);
  }

  pi_result release();

  ~_pi_event();

private:
  // This constructor is private to force programmers to use the make_native /
  // make_user static members in order to create a pi_event for HIP.
  _pi_event(pi_command_type type, pi_context context, pi_queue queue,
            hipStream_t stream, pi_uint32 stream_token);

  pi_command_type commandType_; // The type of command associated with event.

  std::atomic_uint32_t refCount_; // Event reference count.

  bool hasBeenWaitedOn_; // Signifies whether the event has been waited
                         // on through a call to wait(), which implies
                         // that it has completed.

  bool isRecorded_; // Signifies wether a native HIP event has been recorded
                    // yet.
  bool isStarted_;  // Signifies wether the operation associated with the
                    // PI event has started or not
                    //

  pi_uint32 streamToken_;
  pi_uint32 eventId_; // Queue identifier of the event.

  native_type evEnd_; // HIP event handle. If this _pi_event represents a user
                      // event, this will be nullptr.

  native_type evStart_; // HIP event handle associated with the start

  native_type evQueued_; // HIP event handle associated with the time
                         // the command was enqueued

  pi_queue queue_; // pi_queue associated with the event. If this is a user
                   // event, this will be nullptr.

  hipStream_t stream_; // hipStream_t associated with the event. If this is a
                       // user event, this will be uninitialized.

  pi_context context_; // pi_context associated with the event. If this is a
                       // native event, this will be the same context associated
                       // with the queue_ member.
};

/// Implementation of PI Program on HIP Module object
///
struct _pi_program {
  using native_type = hipModule_t;
  native_type module_;
  const char *binary_;
  size_t binarySizeInBytes_;
  std::atomic_uint32_t refCount_;
  _pi_context *context_;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char errorLog_[MAX_LOG_SIZE], infoLog_[MAX_LOG_SIZE];
  std::string buildOptions_;
  pi_program_build_status buildStatus_ = PI_PROGRAM_BUILD_STATUS_NONE;

  _pi_program(pi_context ctxt);
  ~_pi_program();

  pi_result set_binary(const char *binary, size_t binarySizeInBytes);

  pi_result build_program(const char *build_options);

  pi_context get_context() const { return context_; };

  native_type get() const noexcept { return module_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

/// Implementation of a PI Kernel for HIP
///
/// PI Kernels are used to set kernel arguments,
/// creating a state on the Kernel object for a given
/// invocation. This is not the case of HIPFunction objects,
/// which are simply passed together with the arguments on the invocation.
/// The PI Kernel implementation for HIP stores the list of arguments,
/// argument sizes and offsets to emulate the interface of PI Kernel,
/// saving the arguments for the later dispatch.
/// Note that in PI API, the Local memory is specified as a size per
/// individual argument, but in HIP only the total usage of shared
/// memory is required since it is not passed as a parameter.
/// A compiler pass converts the PI API local memory model into the
/// HIP shared model. This object simply calculates the total of
/// shared memory, and the initial offsets of each parameter.
///
struct _pi_kernel {
  using native_type = hipFunction_t;

  native_type function_;
  native_type functionWithOffsetParam_;
  std::string name_;
  pi_context context_;
  pi_program program_;
  std::atomic_uint32_t refCount_;

  /// Structure that holds the arguments to the kernel.
  /// Note earch argument size is known, since it comes
  /// from the kernel signature.
  /// This is not something can be queried from the HIP API
  /// so there is a hard-coded size (\ref MAX_PARAM_BYTES)
  /// and a storage.
  ///
  struct arguments {
    static constexpr size_t MAX_PARAM_BYTES = 4000u;
    using args_t = std::array<char, MAX_PARAM_BYTES>;
    using args_size_t = std::vector<size_t>;
    using args_index_t = std::vector<void *>;
    args_t storage_;
    args_size_t paramSizes_;
    args_index_t indices_;
    args_size_t offsetPerIndex_;

    std::uint32_t implicitOffsetArgs_[3] = {0, 0, 0};

    arguments() {
      // Place the implicit offset index at the end of the indicies collection
      indices_.emplace_back(&implicitOffsetArgs_);
    }

    /// Adds an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    /// Implicit offset argument is kept at the back of the indices collection.
    void add_arg(size_t index, size_t size, const void *arg,
                 size_t localSize = 0) {
      if (index + 2 > indices_.size()) {
        // Move implicit offset argument index with the end
        indices_.resize(index + 2, indices_.back());
        // Ensure enough space for the new argument
        paramSizes_.resize(index + 1);
        offsetPerIndex_.resize(index + 1);
      }
      paramSizes_[index] = size;
      // calculate the insertion point on the array
      size_t insertPos = std::accumulate(std::begin(paramSizes_),
                                         std::begin(paramSizes_) + index, 0);
      // Update the stored value for the argument
      std::memcpy(&storage_[insertPos], arg, size);
      indices_[index] = &storage_[insertPos];
      offsetPerIndex_[index] = localSize;
    }

    void add_local_arg(size_t index, size_t size) {
      size_t localOffset = this->get_local_size();

      // maximum required alignment is the size of the largest vector type
      const size_t max_alignment = sizeof(double) * 16;

      // for arguments smaller than the maximum alignment simply align to the
      // size of the argument
      const size_t alignment = std::min(max_alignment, size);

      // align the argument
      size_t alignedLocalOffset = localOffset;
      if (localOffset % alignment != 0) {
        alignedLocalOffset += alignment - (localOffset % alignment);
      }

      add_arg(index, sizeof(size_t), (const void *)&(alignedLocalOffset),
              size + (alignedLocalOffset - localOffset));
    }

    void set_implicit_offset(size_t size, std::uint32_t *implicitOffset) {
      assert(size == sizeof(std::uint32_t) * 3);
      std::memcpy(implicitOffsetArgs_, implicitOffset, size);
    }

    void clear_local_size() {
      std::fill(std::begin(offsetPerIndex_), std::end(offsetPerIndex_), 0);
    }

    args_index_t get_indices() const noexcept { return indices_; }

    pi_uint32 get_local_size() const {
      return std::accumulate(std::begin(offsetPerIndex_),
                             std::end(offsetPerIndex_), 0);
    }
  } args_;

  _pi_kernel(hipFunction_t func, hipFunction_t funcWithOffsetParam,
             const char *name, pi_program program, pi_context ctxt)
      : function_{func}, functionWithOffsetParam_{funcWithOffsetParam},
        name_{name}, context_{ctxt}, program_{program}, refCount_{1} {
    hip_piProgramRetain(program_);
    pi2ur::piContextRetain(context_);
  }

  _pi_kernel(hipFunction_t func, const char *name, pi_program program,
             pi_context ctxt)
      : _pi_kernel{func, nullptr, name, program, ctxt} {}

  ~_pi_kernel() {
    hip_piProgramRelease(program_);
    pi2ur::piContextRelease(context_);
  }

  pi_program get_program() const noexcept { return program_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  native_type get() const noexcept { return function_; };

  native_type get_with_offset_parameter() const noexcept {
    return functionWithOffsetParam_;
  };

  bool has_with_offset_parameter() const noexcept {
    return functionWithOffsetParam_ != nullptr;
  }

  pi_context get_context() const noexcept { return context_; };

  const char *get_name() const noexcept { return name_.c_str(); }

  /// Returns the number of arguments, excluding the implicit global offset.
  /// Note this only returns the current known number of arguments, not the
  /// real one required by the kernel, since this cannot be queried from
  /// the HIP Driver API
  pi_uint32 get_num_args() const noexcept { return args_.indices_.size() - 1; }

  void set_kernel_arg(int index, size_t size, const void *arg) {
    args_.add_arg(index, size, arg);
  }

  void set_kernel_local_arg(int index, size_t size) {
    args_.add_local_arg(index, size);
  }

  void set_implicit_offset_arg(size_t size, std::uint32_t *implicitOffset) {
    args_.set_implicit_offset(size, implicitOffset);
  }

  arguments::args_index_t get_arg_indices() const {
    return args_.get_indices();
  }

  pi_uint32 get_local_size() const noexcept { return args_.get_local_size(); }

  void clear_local_size() { args_.clear_local_size(); }
};

/// Implementation of samplers for HIP
///
/// Sampler property layout:
/// | 31 30 ... 6 5 |      4 3 2      |     1      |         0        |
/// |      N/A      | addressing mode | fiter mode | normalize coords |
struct _pi_sampler {
  std::atomic_uint32_t refCount_;
  pi_uint32 props_;
  pi_context context_;

  _pi_sampler(pi_context context)
      : refCount_(1), props_(0), context_(context) {}

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

// -------------------------------------------------------------
// Helper types and functions
//

#endif // PI_HIP_HPP

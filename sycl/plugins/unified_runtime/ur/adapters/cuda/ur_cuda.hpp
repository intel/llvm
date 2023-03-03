//===--------- ur_cuda.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

// We need the PI header temporarily while the UR device impl still uses the
// PI context type
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

#include <cuda.h>

// struct _ur_platform_handle_t;
// using ur_platform_handle_t = _ur_platform_handle_t *;
// struct _ur_device_handle_t;
// using ur_device_handle_t = _ur_device_handle_t *;

using ur_stream_guard_ = std::unique_lock<std::mutex>;

struct ur_platform_handle_t_ : public _ur_platform {
  std::vector<std::unique_ptr<ur_device_handle_t_>> devices_;
};

struct ur_device_handle_t_ : public _pi_object {
private:
  using native_type = CUdevice;

  native_type cuDevice_;
  CUcontext cuContext_;
  CUevent evBase_; // CUDA event used as base counter
  std::atomic_uint32_t refCount_;
  ur_platform_handle_t platform_;

  static constexpr pi_uint32 max_work_item_dimensions = 3u;
  size_t max_work_item_sizes[max_work_item_dimensions];
  int max_work_group_size;

public:
  ur_device_handle_t_(native_type cuDevice, CUcontext cuContext, CUevent evBase,
                      ur_platform_handle_t platform)
      : cuDevice_(cuDevice), cuContext_(cuContext),
        evBase_(evBase), refCount_{1}, platform_(platform) {}

  ur_device_handle_t_() { cuDevicePrimaryCtxRelease(cuDevice_); }

  native_type get() const noexcept { return cuDevice_; };

  CUcontext get_context() const noexcept { return cuContext_; };

  uint32_t get_reference_count() const noexcept { return refCount_; }

  ur_platform_handle_t get_platform() const noexcept { return platform_; };

  uint64_t get_elapsed_time(CUevent) const;

  void save_max_work_item_sizes(size_t size,
                                size_t *save_max_work_item_sizes) noexcept {
    memcpy(max_work_item_sizes, save_max_work_item_sizes, size);
  };

  void save_max_work_group_size(int value) noexcept {
    max_work_group_size = value;
  };

  void get_max_work_item_sizes(size_t ret_size,
                               size_t *ret_max_work_item_sizes) const noexcept {
    memcpy(ret_max_work_item_sizes, max_work_item_sizes, ret_size);
  };

  int get_max_work_group_size() const noexcept { return max_work_group_size; };
};

struct ur_context_handle_t_ : _pi_object {

  struct deleter_data {
    pi_context_extended_deleter function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  using native_type = CUcontext;

  native_type cuContext_;
  ur_device_handle_t deviceId_;
  std::atomic_uint32_t refCount_;

  ur_context_handle_t_(ur_device_handle_t_ *devId)
      : cuContext_{devId->get_context()}, deviceId_{devId}, refCount_{1} {
    urDeviceRetain(deviceId_);
  };

  ~ur_context_handle_t_() {
    urDeviceRelease(deviceId_);
  }

  void invoke_extended_deleters() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto &deleter : extended_deleters_) {
      deleter();
    }
  }

  void set_extended_deleter(pi_context_extended_deleter function,
                            void *user_data) {
    std::lock_guard<std::mutex> guard(mutex_);
    extended_deleters_.emplace_back(deleter_data{function, user_data});
  }

  ur_device_handle_t get_device() const noexcept { return deviceId_; }

  native_type get() const noexcept { return cuContext_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
};


/// PI queue mapping on to CUstream objects.
///
struct ur_queue_handle_t_ : _pi_object {

  using native_type = CUstream;
  static constexpr int default_num_compute_streams = 128;
  static constexpr int default_num_transfer_streams = 64;

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
  ur_context_handle_t_ *context_;
  ur_device_handle_t_ *device_;
  // ur_queue_properties_t properties_;
  CUevent barrier_event_ = nullptr;
  CUevent barrier_tmp_event_ = nullptr;
  std::atomic_uint32_t refCount_;
  std::atomic_uint32_t eventCount_;
  std::atomic_uint32_t compute_stream_idx_;
  std::atomic_uint32_t transfer_stream_idx_;
  unsigned int num_compute_streams_;
  unsigned int num_transfer_streams_;
  unsigned int last_sync_compute_streams_;
  unsigned int last_sync_transfer_streams_;
  unsigned int flags_;
  ur_queue_property_t ur_flags_;
  // When compute_stream_sync_mutex_ and compute_stream_mutex_ both need to be
  // locked at the same time, compute_stream_sync_mutex_ should be locked first
  // to avoid deadlocks
  std::mutex compute_stream_sync_mutex_;
  std::mutex compute_stream_mutex_;
  std::mutex transfer_stream_mutex_;
  std::mutex barrier_mutex_;
  bool has_ownership_;

  ur_queue_handle_t_(std::vector<CUstream> &&compute_streams,
                     std::vector<CUstream> &&transfer_streams,
                     ur_context_handle_t_ *context, ur_device_handle_t_ *device,
                     unsigned int flags, ur_queue_property_t ur_flags,
                     bool backend_owns = true)
      : compute_streams_{std::move(compute_streams)},
        transfer_streams_{std::move(transfer_streams)},
        delay_compute_(compute_streams_.size(), false),
        compute_applied_barrier_(compute_streams_.size()),
        transfer_applied_barrier_(transfer_streams_.size()), context_{context},
        device_{device}, refCount_{1}, eventCount_{0},
        compute_stream_idx_{0}, transfer_stream_idx_{0},
        num_compute_streams_{0}, num_transfer_streams_{0},
        last_sync_compute_streams_{0}, last_sync_transfer_streams_{0},
        flags_(flags), ur_flags_(ur_flags), has_ownership_{backend_owns} {
    urContextRetain(context_);
    urDeviceRetain(device_);
  }

  ~ur_queue_handle_t_() {
    urContextRelease(context_);
    urDeviceRelease(device_);
  }

  void compute_stream_wait_for_barrier_if_needed(CUstream stream,
                                                 uint32_t stream_i);
  void transfer_stream_wait_for_barrier_if_needed(CUstream stream,
                                                  uint32_t stream_i);

  // get_next_compute/transfer_stream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type get_next_compute_stream(uint32_t *stream_token = nullptr);
  // this overload tries select a stream that was used by one of dependancies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type get_next_compute_stream(uint32_t num_events_in_wait_list,
                                      const ur_event_handle_t *event_wait_list,
                                      ur_stream_guard_ &guard,
                                      uint32_t *stream_token = nullptr);
  native_type get_next_transfer_stream();
  native_type get() { return get_next_compute_stream(); };

  bool has_been_synchronized(uint32_t stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<pi_uint32>::max()) {
      return false;
    }
    return last_sync_compute_streams_ >= stream_token;
  }

  bool can_reuse_stream(uint32_t stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<uint32_t>::max()) {
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
      std::lock_guard compute_sync_guard(compute_stream_sync_mutex_);
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int start = last_sync_compute_streams_;
      unsigned int end = num_compute_streams_ < size
                             ? num_compute_streams_
                             : compute_stream_idx_.load();
      if (ResetUsed) {
        last_sync_compute_streams_ = end;
      }
      if (end - start >= size) {
        sync_compute(0, size);
      } else {
        start %= size;
        end %= size;
        if (start <= end) {
          sync_compute(start, end);
        } else {
          sync_compute(start, size);
          sync_compute(0, end);
        }
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
        if (ResetUsed) {
          last_sync_transfer_streams_ = end;
        }
        if (end - start >= size) {
          sync_transfer(0, size);
        } else {
          start %= size;
          end %= size;
          if (start <= end) {
            sync_transfer(start, end);
          } else {
            sync_transfer(start, size);
            sync_transfer(0, end);
          }
        }
      }
    }
  }

  ur_context_handle_t_ *get_context() const { return context_; };

  ur_device_handle_t_ *get_device() const { return device_; };

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

  uint32_t get_next_event_id() noexcept { return ++eventCount_; }

  bool backend_has_ownership() const noexcept { return has_ownership_; }
};



/// PI Event mapping to CUevent
///
struct ur_event_handle_t_ : _pi_object {
public:
  using native_type = CUevent;

  ur_result_t record();

  ur_result_t wait();

  ur_result_t start();

  native_type get() const noexcept { return evEnd_; };

  ur_queue_handle_t get_queue() const noexcept { return queue_; }

  CUstream get_stream() const noexcept { return stream_; }

  uint32_t get_compute_stream_token() const noexcept { return streamToken_; }

  ur_command_t get_command_type() const noexcept { return commandType_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

  bool is_recorded() const noexcept { return isRecorded_; }

  bool is_started() const noexcept { return isStarted_; }

  bool is_completed() const noexcept;

  uint32_t get_execution_status() const noexcept {

    if (!is_recorded()) {
      return PI_EVENT_SUBMITTED;
    }

    if (!is_completed()) {
      return PI_EVENT_RUNNING;
    }
    return PI_EVENT_COMPLETE;
  }

  ur_context_handle_t get_context() const noexcept { return context_; };

  uint32_t increment_reference_count() { return ++refCount_; }

  uint32_t decrement_reference_count() { return --refCount_; }

  uint32_t get_event_id() const noexcept { return eventId_; }

  bool backend_has_ownership() const noexcept { return has_ownership_; }

  // Returns the counter time when the associated command(s) were enqueued
  //
  uint64_t get_queued_time() const;

  // Returns the counter time when the associated command(s) started execution
  //
  uint64_t get_start_time() const;

  // Returns the counter time when the associated command(s) completed
  //
  uint64_t get_end_time() const;

  // construct a native CUDA. This maps closely to the underlying CUDA event.
  static ur_event_handle_t
  make_native(ur_command_t type, ur_queue_handle_t queue, CUstream stream,
              uint32_t stream_token = std::numeric_limits<uint32_t>::max()) {
    // TODO(ur): Remove cast when pi_event is ported to UR
    return new ur_event_handle_t_(type,
                         queue->get_context(),
                         queue, stream, stream_token);
  }

  static ur_event_handle_t make_with_native(ur_context_handle_t context, CUevent eventNative) {
    return new ur_event_handle_t_(context, eventNative);
  }

  ur_result_t release();

  ~ur_event_handle_t_();

private:
  // This constructor is private to force programmers to use the make_native /
  // make_user static members in order to create a pi_event for CUDA.
  ur_event_handle_t_(ur_command_t type, ur_context_handle_t context, ur_queue_handle_t queue,
            CUstream stream, uint32_t stream_token);

  // This constructor is private to force programmers to use the
  // make_with_native for event introp
  ur_event_handle_t_(ur_context_handle_t context, CUevent eventNative);

  ur_command_t commandType_; // The type of command associated with event.

  std::atomic_uint32_t refCount_; // Event reference count.

  bool has_ownership_; // Signifies if event owns the native type.

  bool hasBeenWaitedOn_; // Signifies whether the event has been waited
                         // on through a call to wait(), which implies
                         // that it has completed.

  bool isRecorded_; // Signifies wether a native CUDA event has been recorded
                    // yet.
  bool isStarted_;  // Signifies wether the operation associated with the
                    // PI event has started or not
                    //

  uint32_t streamToken_;
  uint32_t eventId_; // Queue identifier of the event.

  native_type evEnd_; // CUDA event handle. If this _pi_event represents a user
                      // event, this will be nullptr.

  native_type evStart_; // CUDA event handle associated with the start

  native_type evQueued_; // CUDA event handle associated with the time
                         // the command was enqueued

  ur_queue_handle_t queue_; // pi_queue associated with the event. If this is a user
                   // event, this will be nullptr.

  CUstream stream_; // CUstream associated with the event. If this is a user
                    // event, this will be uninitialized.

  ur_context_handle_t context_; // pi_context associated with the event. If this is a
                       // native event, this will be the same context associated
                       // with the queue_ member.
};

// Make the Unified Runtime handles definition complete.
// This is used in various "create" API where new handles are allocated.
// struct _zer_platform_handle_t : public _ur_platform_handle_t {
//   using _ur_platform_handle_t::_ur_platform_handle_t;
// };
// struct _zer_device_handle_t : public _ur_device_handle_t {
//   using _ur_device_handle_t::_ur_device_handle_t;
// };

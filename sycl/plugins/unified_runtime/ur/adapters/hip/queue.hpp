//===--------- queue.hpp - HIP Adapter -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"

using ur_stream_quard = std::unique_lock<std::mutex>;

/// UR queue mapping on to hipStream_t objects.
///
struct ur_queue_handle_t_ {
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
  ur_context_handle_t context_;
  ur_device_handle_t device_;
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
  ur_queue_flags_t ur_flags_;
  // When compute_stream_sync_mutex_ and compute_stream_mutex_ both need to be
  // locked at the same time, compute_stream_sync_mutex_ should be locked first
  // to avoid deadlocks
  std::mutex compute_stream_sync_mutex_;
  std::mutex compute_stream_mutex_;
  std::mutex transfer_stream_mutex_;
  std::mutex barrier_mutex_;

  ur_queue_handle_t_(std::vector<native_type> &&compute_streams,
                     std::vector<native_type> &&transfer_streams,
                     ur_context_handle_t context, ur_device_handle_t device,
                     unsigned int flags, ur_queue_flags_t ur_flags)
      : compute_streams_{std::move(compute_streams)},
        transfer_streams_{std::move(transfer_streams)},
        delay_compute_(compute_streams_.size(), false),
        compute_applied_barrier_(compute_streams_.size()),
        transfer_applied_barrier_(transfer_streams_.size()), context_{context},
        device_{device}, refCount_{1}, eventCount_{0}, compute_stream_idx_{0},
        transfer_stream_idx_{0}, num_compute_streams_{0},
        num_transfer_streams_{0}, last_sync_compute_streams_{0},
        last_sync_transfer_streams_{0}, flags_(flags), ur_flags_(ur_flags) {
    urContextRetain(context_);
    urDeviceRetain(device_);
  }

  ~ur_queue_handle_t_() {
    urContextRelease(context_);
    urDeviceRelease(device_);
  }

  void compute_stream_wait_for_barrier_if_needed(hipStream_t stream,
                                                 uint32_t stream_i);
  void transfer_stream_wait_for_barrier_if_needed(hipStream_t stream,
                                                  uint32_t stream_i);

  // get_next_compute/transfer_stream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type get_next_compute_stream(uint32_t *stream_token = nullptr);
  // this overload tries select a stream that was used by one of dependancies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type get_next_compute_stream(uint32_t num_events_in_wait_list,
                                      const ur_event_handle_t *event_wait_list,
                                      ur_stream_quard &guard,
                                      uint32_t *stream_token = nullptr);
  native_type get_next_transfer_stream();
  native_type get() { return get_next_compute_stream(); };

  bool has_been_synchronized(uint32_t stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    return last_sync_compute_streams_ > stream_token;
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

  ur_context_handle_t get_context() const { return context_; };

  ur_device_handle_t get_device() const { return device_; };

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

  uint32_t get_next_event_id() noexcept { return ++eventCount_; }
};

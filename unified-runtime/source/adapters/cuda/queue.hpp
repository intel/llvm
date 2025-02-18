//===--------- queue.hpp - CUDA Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include <ur/ur.hpp>

#include <algorithm>
#include <cuda.h>
#include <mutex>
#include <vector>

using ur_stream_guard_ = std::unique_lock<std::mutex>;

/// UR queue mapping on to CUstream objects.
///
struct ur_queue_handle_t_ {

  using native_type = CUstream;
  static constexpr int DefaultNumComputeStreams = 128;
  static constexpr int DefaultNumTransferStreams = 64;

  std::vector<native_type> ComputeStreams;
  std::vector<native_type> TransferStreams;
  // Stream used for recording EvQueue, which holds information about when the
  // command in question is enqueued on host, as opposed to started. It is
  // created only if profiling is enabled - either for queue or per event.
  native_type HostSubmitTimeStream{0};
  // delay_compute_ keeps track of which streams have been recently reused and
  // their next use should be delayed. If a stream has been recently reused it
  // will be skipped the next time it would be selected round-robin style. When
  // skipped, its delay flag is cleared.
  std::vector<bool> DelayCompute;
  // keep track of which streams have applied barrier
  std::vector<bool> ComputeAppliedBarrier;
  std::vector<bool> TransferAppliedBarrier;
  ur_context_handle_t_ *Context;
  ur_device_handle_t_ *Device;
  CUevent BarrierEvent = nullptr;
  CUevent BarrierTmpEvent = nullptr;
  std::atomic_uint32_t RefCount;
  std::atomic_uint32_t EventCount;
  std::atomic_uint32_t ComputeStreamIndex;
  std::atomic_uint32_t TransferStreamIndex;
  unsigned int NumComputeStreams;
  unsigned int NumTransferStreams;
  unsigned int LastSyncComputeStreams;
  unsigned int LastSyncTransferStreams;
  unsigned int Flags;
  ur_queue_flags_t URFlags;
  int Priority;
  // When ComputeStreamSyncMutex and ComputeStreamMutex both need to be
  // locked at the same time, ComputeStreamSyncMutex should be locked first
  // to avoid deadlocks
  std::mutex ComputeStreamSyncMutex;
  std::mutex ComputeStreamMutex;
  std::mutex TransferStreamMutex;
  std::mutex BarrierMutex;
  bool HasOwnership;

  ur_queue_handle_t_(std::vector<CUstream> &&ComputeStreams,
                     std::vector<CUstream> &&TransferStreams,
                     ur_context_handle_t_ *Context, ur_device_handle_t_ *Device,
                     unsigned int Flags, ur_queue_flags_t URFlags, int Priority,
                     bool BackendOwns = true)
      : ComputeStreams{std::move(ComputeStreams)},
        TransferStreams{std::move(TransferStreams)},
        DelayCompute(this->ComputeStreams.size(), false),
        ComputeAppliedBarrier(this->ComputeStreams.size()),
        TransferAppliedBarrier(this->TransferStreams.size()), Context{Context},
        Device{Device}, RefCount{1}, EventCount{0}, ComputeStreamIndex{0},
        TransferStreamIndex{0}, NumComputeStreams{0}, NumTransferStreams{0},
        LastSyncComputeStreams{0}, LastSyncTransferStreams{0}, Flags(Flags),
        URFlags(URFlags), Priority(Priority), HasOwnership{BackendOwns} {
    urContextRetain(Context);
    urDeviceRetain(Device);
  }

  ~ur_queue_handle_t_() {
    urContextRelease(Context);
    urDeviceRelease(Device);
  }

  void computeStreamWaitForBarrierIfNeeded(CUstream Strean, uint32_t StreamI);
  void transferStreamWaitForBarrierIfNeeded(CUstream Stream, uint32_t StreamI);

  // get_next_compute/transfer_stream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type getNextComputeStream(uint32_t *StreamToken = nullptr);
  // this overload tries select a stream that was used by one of dependencies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type getNextComputeStream(uint32_t NumEventsInWaitList,
                                   const ur_event_handle_t *EventWaitList,
                                   ur_stream_guard_ &Guard,
                                   uint32_t *StreamToken = nullptr);

  // Thread local stream will be used if ScopedStream is active
  static CUstream &getThreadLocalStream() {
    static thread_local CUstream stream{0};
    return stream;
  }

  native_type getNextTransferStream();
  native_type get() { return getNextComputeStream(); };
  ur_device_handle_t getDevice() const noexcept { return Device; };

  // Function which creates the profiling stream. Called only from makeNative
  // event when profiling is required.
  void createHostSubmitTimeStream() {
    static std::once_flag HostSubmitTimeStreamFlag;
    std::call_once(HostSubmitTimeStreamFlag, [&]() {
      UR_CHECK_ERROR(cuStreamCreateWithPriority(&HostSubmitTimeStream,
                                                CU_STREAM_NON_BLOCKING, 0));
    });
  }

  native_type getHostSubmitTimeStream() { return HostSubmitTimeStream; }

  bool hasBeenSynchronized(uint32_t StreamToken) {
    // stream token not associated with one of the compute streams
    if (StreamToken == std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    return LastSyncComputeStreams > StreamToken;
  }

  bool canReuseStream(uint32_t StreamToken) {
    // stream token not associated with one of the compute streams
    if (StreamToken == std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    // If the command represented by the stream token was not the last command
    // enqueued to the stream we can not reuse the stream - we need to allow for
    // commands enqueued after it and the one we are about to enqueue to run
    // concurrently
    bool IsLastCommand =
        (ComputeStreamIndex - StreamToken) <= ComputeStreams.size();
    // If there was a barrier enqueued to the queue after the command
    // represented by the stream token we should not reuse the stream, as we can
    // not take that stream into account for the bookkeeping for the next
    // barrier - such a stream would not be synchronized with. Performance-wise
    // it does not matter that we do not reuse the stream, as the work
    // represented by the stream token is guaranteed to be complete by the
    // barrier before any work we are about to enqueue to the stream will start,
    // so the event does not need to be synchronized with.
    return IsLastCommand && !hasBeenSynchronized(StreamToken);
  }

  template <typename T> bool allOf(T &&F) {
    {
      std::lock_guard<std::mutex> ComputeGuard(ComputeStreamMutex);
      unsigned int End = std::min(
          static_cast<unsigned int>(ComputeStreams.size()), NumComputeStreams);
      if (!std::all_of(ComputeStreams.begin(), ComputeStreams.begin() + End, F))
        return false;
    }
    {
      std::lock_guard<std::mutex> TransferGuard(TransferStreamMutex);
      unsigned int End =
          std::min(static_cast<unsigned int>(TransferStreams.size()),
                   NumTransferStreams);
      if (!std::all_of(TransferStreams.begin(), TransferStreams.begin() + End,
                       F))
        return false;
    }
    return true;
  }

  template <typename T> void forEachStream(T &&F) {
    {
      std::lock_guard<std::mutex> compute_guard(ComputeStreamMutex);
      unsigned int End = std::min(
          static_cast<unsigned int>(ComputeStreams.size()), NumComputeStreams);
      for (unsigned int i = 0; i < End; i++) {
        F(ComputeStreams[i]);
      }
    }
    {
      std::lock_guard<std::mutex> transfer_guard(TransferStreamMutex);
      unsigned int End =
          std::min(static_cast<unsigned int>(TransferStreams.size()),
                   NumTransferStreams);
      for (unsigned int i = 0; i < End; i++) {
        F(TransferStreams[i]);
      }
    }
  }

  template <bool ResetUsed = false, typename T> void syncStreams(T &&F) {
    auto SyncCompute = [&F, &Streams = ComputeStreams, &Delay = DelayCompute](
                           unsigned int Start, unsigned int Stop) {
      for (unsigned int i = Start; i < Stop; i++) {
        F(Streams[i]);
        Delay[i] = false;
      }
    };
    auto SyncTransfer = [&F, &streams = TransferStreams](unsigned int Start,
                                                         unsigned int Stop) {
      for (unsigned int i = Start; i < Stop; i++) {
        F(streams[i]);
      }
    };
    {
      unsigned int Size = static_cast<unsigned int>(ComputeStreams.size());
      std::lock_guard<std::mutex> ComputeSyncGuard(ComputeStreamSyncMutex);
      std::lock_guard<std::mutex> ComputeGuard(ComputeStreamMutex);
      unsigned int Start = LastSyncComputeStreams;
      unsigned int End = NumComputeStreams < Size ? NumComputeStreams
                                                  : ComputeStreamIndex.load();
      if (ResetUsed) {
        LastSyncComputeStreams = End;
      }
      if (End - Start >= Size) {
        SyncCompute(0, Size);
      } else {
        Start %= Size;
        End %= Size;
        if (Start <= End) {
          SyncCompute(Start, End);
        } else {
          SyncCompute(Start, Size);
          SyncCompute(0, End);
        }
      }
    }
    {
      unsigned int Size = static_cast<unsigned int>(TransferStreams.size());
      if (!Size) {
        return;
      }
      std::lock_guard<std::mutex> TransferGuard(TransferStreamMutex);
      unsigned int Start = LastSyncTransferStreams;
      unsigned int End = NumTransferStreams < Size ? NumTransferStreams
                                                   : TransferStreamIndex.load();
      if (ResetUsed) {
        LastSyncTransferStreams = End;
      }
      if (End - Start >= Size) {
        SyncTransfer(0, Size);
      } else {
        Start %= Size;
        End %= Size;
        if (Start <= End) {
          SyncTransfer(Start, End);
        } else {
          SyncTransfer(Start, Size);
          SyncTransfer(0, End);
        }
      }
    }
  }

  ur_context_handle_t_ *getContext() const { return Context; };

  ur_device_handle_t_ *get_device() const { return Device; };

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  uint32_t getNextEventID() noexcept { return ++EventCount; }

  bool backendHasOwnership() const noexcept { return HasOwnership; }
};

// RAII object to make hQueue stream getter methods all return the same stream
// within the lifetime of this object.
//
// This is useful for urEnqueueNativeCommandExp where we want guarantees that
// the user submitted native calls will be dispatched to a known stream, which
// must be "got" within the user submitted fuction.
class ScopedStream {
  ur_queue_handle_t hQueue;

public:
  ScopedStream(ur_queue_handle_t hQueue, uint32_t NumEventsInWaitList,
               const ur_event_handle_t *EventWaitList)
      : hQueue{hQueue} {
    ur_stream_guard_ Guard;
    hQueue->getThreadLocalStream() =
        hQueue->getNextComputeStream(NumEventsInWaitList, EventWaitList, Guard);
  }
  CUstream getStream() { return hQueue->getThreadLocalStream(); }
  ~ScopedStream() { hQueue->getThreadLocalStream() = CUstream{0}; }
};

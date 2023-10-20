//===--------- queue.hpp - HIP Adapter ------------------------------------===//
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

using ur_stream_quard = std::unique_lock<std::mutex>;

/// UR queue mapping on to hipStream_t objects.
///
struct ur_queue_handle_t_ {
  using native_type = hipStream_t;
  static constexpr int DefaultNumComputeStreams = 64;
  static constexpr int DefaultNumTransferStreams = 16;

  std::vector<native_type> ComputeStreams;
  std::vector<native_type> TransferStreams;
  // DelayCompute keeps track of which streams have been recently reused and
  // their next use should be delayed. If a stream has been recently reused it
  // will be skipped the next time it would be selected round-robin style. When
  // skipped, its delay flag is cleared.
  std::vector<bool> DelayCompute;
  // keep track of which streams have applied barrier
  std::vector<bool> ComputeAppliedBarrier;
  std::vector<bool> TransferAppliedBarrier;
  ur_context_handle_t Context;
  ur_device_handle_t Device;
  hipEvent_t BarrierEvent = nullptr;
  hipEvent_t BarrierTmpEvent = nullptr;
  std::atomic_uint32_t RefCount;
  std::atomic_uint32_t EventCount;
  std::atomic_uint32_t ComputeStreamIdx;
  std::atomic_uint32_t TransferStreamIdx;
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

  ur_queue_handle_t_(std::vector<native_type> &&ComputeStreams,
                     std::vector<native_type> &&TransferStreams,
                     ur_context_handle_t Context, ur_device_handle_t Device,
                     unsigned int Flags, ur_queue_flags_t URFlags, int Priority,
                     bool BackendOwns = true)
      : ComputeStreams{std::move(ComputeStreams)}, TransferStreams{std::move(
                                                       TransferStreams)},
        DelayCompute(this->ComputeStreams.size(), false),
        ComputeAppliedBarrier(this->ComputeStreams.size()),
        TransferAppliedBarrier(this->TransferStreams.size()), Context{Context},
        Device{Device}, RefCount{1}, EventCount{0}, ComputeStreamIdx{0},
        TransferStreamIdx{0}, NumComputeStreams{0}, NumTransferStreams{0},
        LastSyncComputeStreams{0}, LastSyncTransferStreams{0}, Flags(Flags),
        URFlags(URFlags), Priority(Priority), HasOwnership{BackendOwns} {
    urContextRetain(Context);
    urDeviceRetain(Device);
  }

  ~ur_queue_handle_t_() {
    urContextRelease(Context);
    urDeviceRelease(Device);
  }

  void computeStreamWaitForBarrierIfNeeded(hipStream_t Stream,
                                           uint32_t Stream_i);
  void transferStreamWaitForBarrierIfNeeded(hipStream_t Stream,
                                            uint32_t Stream_i);

  // getNextCompute/TransferStream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type getNextComputeStream(uint32_t *StreamToken = nullptr);
  // this overload tries select a stream that was used by one of dependencies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type getNextComputeStream(uint32_t NumEventsInWaitList,
                                   const ur_event_handle_t *EventWaitList,
                                   ur_stream_quard &Guard,
                                   uint32_t *StreamToken = nullptr);
  native_type getNextTransferStream();
  native_type get() { return getNextComputeStream(); };

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
        (ComputeStreamIdx - StreamToken) <= ComputeStreams.size();
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
      std::lock_guard<std::mutex> ComputeGuard(ComputeStreamMutex);
      unsigned int End = std::min(
          static_cast<unsigned int>(ComputeStreams.size()), NumComputeStreams);
      for (unsigned int i = 0; i < End; i++) {
        F(ComputeStreams[i]);
      }
    }
    {
      std::lock_guard<std::mutex> TransferGuard(TransferStreamMutex);
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
    auto SyncTransfer = [&F, &Streams = TransferStreams](unsigned int Start,
                                                         unsigned int Stop) {
      for (unsigned int i = Start; i < Stop; i++) {
        F(Streams[i]);
      }
    };
    {
      unsigned int Size = static_cast<unsigned int>(ComputeStreams.size());
      std::lock_guard<std::mutex> ComputeSyncGuard(ComputeStreamSyncMutex);
      std::lock_guard<std::mutex> ComputeGuard(ComputeStreamMutex);
      unsigned int Start = LastSyncComputeStreams;
      unsigned int End = NumComputeStreams < Size ? NumComputeStreams
                                                  : ComputeStreamIdx.load();
      if (End - Start >= Size) {
        SyncCompute(0, Size);
      } else {
        Start %= Size;
        End %= Size;
        if (Start < End) {
          SyncCompute(Start, End);
        } else {
          SyncCompute(Start, Size);
          SyncCompute(0, End);
        }
      }
      if (ResetUsed) {
        LastSyncComputeStreams = End;
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
                                                   : TransferStreamIdx.load();
      if (End - Start >= Size) {
        SyncTransfer(0, Size);
      } else {
        Start %= Size;
        End %= Size;
        if (Start < End) {
          SyncTransfer(Start, End);
        } else {
          SyncTransfer(Start, Size);
          SyncTransfer(0, End);
        }
      }
      if (ResetUsed) {
        LastSyncTransferStreams = End;
      }
    }
  }

  ur_context_handle_t getContext() const { return Context; };

  ur_device_handle_t getDevice() const { return Device; };

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  uint32_t getNextEventId() noexcept { return ++EventCount; }

  bool backendHasOwnership() const noexcept { return HasOwnership; }
};

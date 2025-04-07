/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

using ur_stream_guard = std::unique_lock<std::mutex>;

/// Generic implementation of an out-of-order UR queue based on in-order
/// backend 'stream' objects.
///
/// This class is specifically designed for the CUDA and HIP adapters.
template <typename ST, int CS, int TS> struct stream_queue_t {
  using native_type = ST;
  static constexpr int DefaultNumComputeStreams = CS;
  static constexpr int DefaultNumTransferStreams = TS;

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
  std::atomic_uint32_t RefCount{1};
  std::atomic_uint32_t EventCount{0};
  std::atomic_uint32_t ComputeStreamIndex{0};
  std::atomic_uint32_t TransferStreamIndex{0};
  unsigned int NumComputeStreams{0};
  unsigned int NumTransferStreams{0};
  unsigned int LastSyncComputeStreams{0};
  unsigned int LastSyncTransferStreams{0};
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

  stream_queue_t(bool IsOutOfOrder, ur_context_handle_t_ *Context,
                 ur_device_handle_t_ *Device, unsigned int Flags,
                 ur_queue_flags_t URFlags, int Priority)
      : ComputeStreams(IsOutOfOrder ? DefaultNumComputeStreams : 1),
        TransferStreams(IsOutOfOrder ? DefaultNumTransferStreams : 0),
        DelayCompute(this->ComputeStreams.size(), false),
        ComputeAppliedBarrier(this->ComputeStreams.size()),
        TransferAppliedBarrier(this->TransferStreams.size()), Context{Context},
        Device{Device}, Flags(Flags), URFlags(URFlags), Priority(Priority),
        HasOwnership{true} {
    urContextRetain(Context);
  }

  // Create a queue from a native handle
  stream_queue_t(native_type stream, ur_context_handle_t_ *Context,
                 ur_device_handle_t_ *Device, unsigned int Flags,
                 ur_queue_flags_t URFlags, bool BackendOwns)
      : ComputeStreams(1, stream), TransferStreams(0),
        DelayCompute(this->ComputeStreams.size(), false),
        ComputeAppliedBarrier(this->ComputeStreams.size()),
        TransferAppliedBarrier(this->TransferStreams.size()), Context{Context},
        Device{Device}, NumComputeStreams{1}, Flags(Flags), URFlags(URFlags),
        Priority(0), HasOwnership{BackendOwns} {
    urContextRetain(Context);
  }

  virtual ~stream_queue_t() { urContextRelease(Context); }

  virtual void computeStreamWaitForBarrierIfNeeded(native_type Strean,
                                                   uint32_t StreamI) = 0;
  virtual void transferStreamWaitForBarrierIfNeeded(native_type Stream,
                                                    uint32_t StreamI) = 0;
  virtual void createStreamWithPriority(native_type *Stream, unsigned int Flags,
                                        int Priority) = 0;
  virtual ur_queue_handle_t getEventQueue(const ur_event_handle_t) = 0;
  virtual uint32_t getEventComputeStreamToken(const ur_event_handle_t) = 0;
  virtual native_type getEventStream(const ur_event_handle_t) = 0;

  // get_next_compute/transfer_stream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type getNextComputeStream(uint32_t *StreamToken = nullptr) {
    if (getThreadLocalStream() != native_type{0})
      return getThreadLocalStream();
    uint32_t StreamI;
    uint32_t Token;
    while (true) {
      if (NumComputeStreams < ComputeStreams.size()) {
        // the check above is for performance - so as not to lock mutex every
        // time
        std::lock_guard<std::mutex> guard(ComputeStreamMutex);
        // The second check is done after mutex is locked so other threads can
        // not change NumComputeStreams after that
        if (NumComputeStreams < ComputeStreams.size()) {
          createStreamWithPriority(&ComputeStreams[NumComputeStreams], Flags,
                                   Priority);
          ++NumComputeStreams;
        }
      }
      Token = ComputeStreamIndex++;
      StreamI = Token % ComputeStreams.size();
      // if a stream has been reused before it was next selected round-robin
      // fashion, we want to delay its next use and instead select another one
      // that is more likely to have completed all the enqueued work.
      if (DelayCompute[StreamI]) {
        DelayCompute[StreamI] = false;
      } else {
        break;
      }
    }
    if (StreamToken) {
      *StreamToken = Token;
    }
    native_type res = ComputeStreams[StreamI];
    computeStreamWaitForBarrierIfNeeded(res, StreamI);
    return res;
  }

  // this overload tries select a stream that was used by one of dependencies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type getNextComputeStream(uint32_t NumEventsInWaitList,
                                   const ur_event_handle_t *EventWaitList,
                                   ur_stream_guard &Guard,
                                   uint32_t *StreamToken = nullptr) {
    if (getThreadLocalStream() != native_type{0})
      return getThreadLocalStream();
    for (uint32_t i = 0; i < NumEventsInWaitList; i++) {
      uint32_t Token = getEventComputeStreamToken(EventWaitList[i]);
      if (getEventQueue(EventWaitList[i]) == this && canReuseStream(Token)) {
        std::unique_lock<std::mutex> ComputeSyncGuard(ComputeStreamSyncMutex);
        // redo the check after lock to avoid data races on
        // LastSyncComputeStreams
        if (canReuseStream(Token)) {
          uint32_t StreamI = Token % DelayCompute.size();
          DelayCompute[StreamI] = true;
          if (StreamToken) {
            *StreamToken = Token;
          }
          Guard = ur_stream_guard{std::move(ComputeSyncGuard)};
          native_type Result = getEventStream(EventWaitList[i]);
          computeStreamWaitForBarrierIfNeeded(Result, StreamI);
          return Result;
        }
      }
    }
    Guard = {};
    return getNextComputeStream(StreamToken);
  }

  // Thread local stream will be used if ScopedStream is active
  static native_type &getThreadLocalStream() {
    static thread_local native_type stream{0};
    return stream;
  }

  native_type getNextTransferStream() {
    if (getThreadLocalStream() != native_type{0})
      return getThreadLocalStream();
    if (TransferStreams.empty()) { // for example in in-order queue
      return getNextComputeStream();
    }
    if (NumTransferStreams < TransferStreams.size()) {
      // the check above is for performance - so as not to lock mutex every time
      std::lock_guard<std::mutex> Guard(TransferStreamMutex);
      // The second check is done after mutex is locked so other threads can not
      // change NumTransferStreams after that
      if (NumTransferStreams < TransferStreams.size()) {
        createStreamWithPriority(&TransferStreams[NumTransferStreams], Flags,
                                 Priority);
        ++NumTransferStreams;
      }
    }
    uint32_t StreamI = TransferStreamIndex++ % TransferStreams.size();
    native_type Result = TransferStreams[StreamI];
    transferStreamWaitForBarrierIfNeeded(Result, StreamI);
    return Result;
  }

  native_type get() { return getNextComputeStream(); };
  ur_device_handle_t getDevice() const noexcept { return Device; };

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

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  uint32_t getNextEventId() noexcept { return ++EventCount; }

  bool backendHasOwnership() const noexcept { return HasOwnership; }
};

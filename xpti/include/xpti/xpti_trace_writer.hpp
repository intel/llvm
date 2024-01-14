//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include "xpti/xpti_data_types.h"
#include "xpti/xpti_spin_lock.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace xpti {
namespace trace {

/// A generic profiling trace writer.
class TraceWriter {
public:
  void addBeginEvent(const char *EventName,
                     std::initializer_list<const char *> Categories,
                     uint64_t Instance, trace_event_data_t *Event,
                     size_t PID = 0, size_t TID = 0, size_t TimeStamp = 0) {
    saveTracePoint(Type::Begin, EventName, std::move(Categories), Instance,
                   /*Value=*/0, /*Parent=*/nullptr, Event, PID, TID, TimeStamp);
  }
  void addEndEvent(const char *EventName,
                   std::initializer_list<const char *> Categories,
                   uint64_t Instance, trace_event_data_t *Event, size_t PID = 0,
                   size_t TID = 0, size_t TimeStamp = 0) {
    saveTracePoint(Type::End, EventName, std::move(Categories), Instance,
                   /*Value=*/0, /*Parent=*/nullptr, Event, PID, TID, TimeStamp);
  }
  void addCounterEvent(const char *EventName,
                       std::initializer_list<const char *> Categories,
                       uint64_t Instance, uint64_t Value,
                       trace_event_data_t *Event, size_t PID = 0,
                       size_t TID = 0, size_t TimeStamp = 0) {
    saveTracePoint(Type::Counter, EventName, std::move(Categories), Instance,
                   Value, /*Parent=*/nullptr, Event, PID, TID, TimeStamp);
  }
  void addInstantEvent(const char *EventName,
                       std::initializer_list<const char *> Categories,
                       uint64_t Instance, trace_event_data_t *Event,
                       size_t PID = 0, size_t TID = 0, size_t TimeStamp = 0) {
    saveTracePoint(Type::Instant, EventName, std::move(Categories), Instance,
                   /*Value=*/0, /*Parent=*/nullptr, Event, PID, TID, TimeStamp);
  }

  void connectEvents(trace_event_data_t *Parent, trace_event_data_t *Event) {
    saveTracePoint(
        Type::Connect, /*EventName=*/nullptr, /*Catecories=*/{}, /*Instance=*/0,
        /*Value=*/0, Parent, Event, /*PID=*/0, /*TID=*/0, /*TimeStamp=*/0);
  }

  virtual ~TraceWriter() = default;

protected:
  enum class Type {
    Begin,
    End,
    Counter,
    Instant,
    Connect,
  };

  struct TracePoint {
    Type TracePointType;
    const char *EventName;
    uint64_t Instance;
    size_t NumCategories;
    std::array<const char *, 8> Categories;
    trace_event_data_t *Parent;
    trace_event_data_t *Event;
    uint64_t Value;
    size_t PID;
    size_t TID;
    size_t TimeStamp;
  };

  class TracePointBuffer {
  public:
    constexpr static size_t BUFFER_SIZE = 256;
    TracePointBuffer() { MTracePoints.reserve(BUFFER_SIZE); }

    bool isFull() const noexcept { return MTracePoints.size() == BUFFER_SIZE; }

    void add(TracePoint &&tp) noexcept {
      MTracePoints.push_back(std::forward<TracePoint>(tp));
    }

    std::vector<TracePoint>::const_iterator begin() const noexcept {
      return MTracePoints.begin();
    }

    std::vector<TracePoint>::const_iterator end() const noexcept {
      return MTracePoints.end();
    }

  private:
    std::vector<TracePoint> MTracePoints;
  };

  TracePointBuffer *getOrCreateBuffer() {
    SharedLock<SharedSpinLock> Lock{MBuffersMutex};
    auto ID = std::this_thread::get_id();

    if (MBuffers.count(ID) == 0) {
      Lock.upgrade_to_writer();
      MBuffers.insert({ID, TracePointBuffer()});
    }

    return &MBuffers.at(ID);
  }

  void submitCurrentBuffer() {
    SharedLock<SharedSpinLock> Lock{MBuffersMutex};
    auto ID = std::this_thread::get_id();
    {
      std::unique_lock<std::mutex> QueueLock{MQueueMutex};
      MBufferQueue.push(std::move(MBuffers.at(ID)));
      MQueueCV.notify_one();
    }
    MBuffers[ID] = std::move(TracePointBuffer());
  }

  void saveTracePoint(Type Ty, const char *EventName,
                      std::initializer_list<const char *> Categories,
                      uint64_t Instance, uint64_t Value,
                      trace_event_data_t *Parent, trace_event_data_t *Event,
                      size_t PID, size_t TID, size_t TimeStamp) {
    TracePointBuffer *Buffer = getOrCreateBuffer();
    if (Buffer->isFull()) {
      submitCurrentBuffer();
      Buffer = getOrCreateBuffer();
    }

    TracePoint TP;
    TP.TracePointType = Ty;
    TP.EventName = EventName;
    TP.Instance = Instance;
    TP.NumCategories = Categories.size();
    TP.Parent = Parent;
    TP.Event = Event;
    TP.PID = PID;
    TP.TID = TID;
    TP.TimeStamp = TimeStamp;

    size_t I = 0;
    for (auto Cat : Categories)
      TP.Categories[I++] = Cat;

    Buffer->add(std::move(TP));
  }

  std::mutex MQueueMutex;
  std::queue<TracePointBuffer> MBufferQueue;
  std::condition_variable MQueueCV;

  SharedSpinLock MBuffersMutex;
  std::unordered_map<std::thread::id, TracePointBuffer> MBuffers;
};

/// Chrome JSON tracing writer.
class JSONWriter : public TraceWriter {
  using Super = TraceWriter;

public:
  JSONWriter(const std::string &Path) {
    MFile = std::fopen(Path.c_str(), "w");
    write("{\n");
    write("  \"traceEvents\": [\n");
    MWorker = std::thread([this] { run(); });
  }

  ~JSONWriter() {
    MIsRunning.store(false);
    MWorker.join();

    for (const auto &ThreadData : Super::MBuffers) {
      for (const auto &TP : ThreadData.second)
        encodeTracepoint(TP);
    }
    std::fflush(MFile);

    write("    {\"name\":\"\", \"cat\":\"\", \"ph\":\"\", \"pid\":\"\", "
          "\"tid\":\"\", \"ts\":\"\"}\n");
    write("  ],\n");
    write("  \"displayTimeUnit\":\"ns\"\n}\n");
    std::fclose(MFile);
  }

private:
  void run() {
    while (MIsRunning.load() != false) {
      std::unique_lock<std::mutex> Lock{Super::MQueueMutex};
      using namespace std::literals;
      // FIXME: the correct and efficient way to do wait for new buffers in
      // queue is to use std::stop_token with condition variables.
      // Unfortunately, that's a C++20 feature. Instead, wait for a small amount
      // of time to wake and check if we have to stop the execution.
      Super::MQueueCV.wait_for(
          Lock, 1ms, [&, this] { return !Super::MBufferQueue.empty(); });

      if (MIsRunning.load() == false)
        break;

      if (MBufferQueue.empty())
        continue;

      Super::TracePointBuffer Buffer = std::move(Super::MBufferQueue.front());
      Super::MBufferQueue.pop();
      Lock.unlock();

      for (const auto &TP : Buffer)
        encodeTracepoint(TP);

      std::fflush(MFile);
    }
  }

  void write(const char *Str) {
    std::fwrite(Str, sizeof(char), std::strlen(Str), MFile);
  }

  void encodeTracepoint(const Super::TracePoint &TP) {
    // Not supported in JSON format for now
    if (TP.TracePointType == Super::Type::Connect)
      return;

    write("    {\"name\": \"");
    write(TP.EventName);

    write("\", \"cat\": \"");
    for (size_t i = 0; i < TP.NumCategories - 1; i++) {
      write(TP.Categories[i]);
      write(",");
    }
    write(TP.Categories[TP.NumCategories - 1]);
    write("\", ");

    switch (TP.TracePointType) {
    case Super::Type::Begin:
      write("\"ph\": \"B\", ");
      break;
    case Super::Type::End:
      write("\"ph\": \"E\", ");
      break;
    case Super::Type::Counter:
      write("\"ph\": \"C\", ");
      std::fprintf(MFile, "\"args\": {\"value\": %lu}, ", TP.Value);
      break;
    case Super::Type::Instant:
      write("\"ph\": \"i\", ");
      break;
    default:
      break;
    }

    std::fprintf(MFile, "\"pid\": %lu, ", TP.PID);
    std::fprintf(MFile, "\"tid\": %lu, ", TP.TID);
    std::fprintf(MFile, "\"ts\": %lu", TP.TimeStamp);

    write("},\n");
  }

  std::FILE *MFile;
  std::atomic_bool MIsRunning = true;
  std::thread MWorker;
};
} // namespace trace
} // namespace xpti

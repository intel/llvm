//
// An example collector/tool that prints out just the SYCL PI layer trace
// events
//
#include "xpti/xpti_trace_framework.h"
#include "xpti_timers.hpp"
#include "xpti_writers.hpp"

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

std::set<std::string> GStreamsToObserve;
xpti::writer *GWriter = nullptr;

uint64_t GProcessID = 0;

using incomplete_records_t = std::unordered_map<uint64_t, xpti::record_t>;
using uid_lookup_t = std::unordered_map<uint64_t, incomplete_records_t>;
using stream_records_t = std::unordered_map<std::string, uid_lookup_t>;
stream_records_t GIncompleteRecords;
std::mutex GRecMutex, GStreamMutex;

static void record_state(xpti::record_t &r, bool begin_scope) {
  xpti::utils::timer::measurement_t m;
  if (begin_scope) {
    r.TSBegin = m.clock();
    r.Flags |= (uint64_t)(xpti::RecordFlags::BeginTimePresent);
    r.HWID = m.cpu();
    r.TID = m.thread();
    r.PID = GProcessID;
    r.Flags |= ((uint64_t)xpti::RecordFlags::TimeInNanoseconds);
  } else {
    r.TSEnd = m.clock();
    r.Flags |= (uint64_t)(xpti::RecordFlags::EndTimePresent);
    uint64_t mask = (uint64_t)xpti::RecordFlags::InvalidRecord;
    mask = ~(mask);
    r.Flags &= mask;
    r.Flags |= ((uint64_t)xpti::RecordFlags::ValidRecord);
  }
}

XPTI_CALLBACK_API void traceCallback(uint16_t trace_type,
                                     xpti::trace_event_data_t *parent,
                                     xpti::trace_event_data_t *event,
                                     uint64_t instance, const void *user_data);
XPTI_CALLBACK_API void graphCallback(uint16_t trace_type,
                                     xpti::trace_event_data_t *parent,
                                     xpti::trace_event_data_t *event,
                                     uint64_t instance, const void *user_data);
XPTI_CALLBACK_API void syclPiCallback(uint16_t trace_type,
                                      xpti::trace_event_data_t *parent,
                                      xpti::trace_event_data_t *event,
                                      uint64_t instance, const void *user_data);
XPTI_CALLBACK_API void syclPerfCallback(uint16_t trace_type,
                                        xpti::trace_event_data_t *parent,
                                        xpti::trace_event_data_t *event,
                                        uint64_t instance,
                                        const void *user_data);
XPTI_CALLBACK_API void syclMemCallback(uint16_t trace_type,
                                       xpti::trace_event_data_t *parent,
                                       xpti::trace_event_data_t *event,
                                       uint64_t instance,
                                       const void *user_data);
XPTI_CALLBACK_API void syclL0Callback(uint16_t trace_type,
                                      xpti::trace_event_data_t *parent,
                                      xpti::trace_event_data_t *event,
                                      uint64_t instance, const void *user_data);
XPTI_CALLBACK_API void syclCudaCallback(uint16_t trace_type,
                                        xpti::trace_event_data_t *parent,
                                        xpti::trace_event_data_t *event,
                                        uint64_t instance,
                                        const void *user_data);
XPTI_CALLBACK_API void graphMemCallback(uint16_t trace_type,
                                        xpti::trace_event_data_t *parent,
                                        xpti::trace_event_data_t *event,
                                        uint64_t instance,
                                        const void *user_data);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version,
                                     const char *version_str,
                                     const char *stream_name) {
  // Eventually, we can expose the streams to observe through an environment
  // variable and manage it, if required; currently provides the control we need
  // with code changes
  static bool InitStreams = true;
  if (InitStreams) {
    GStreamsToObserve.insert("sycl");
    GStreamsToObserve.insert("sycl.experimental.mem.alloc");
    GStreamsToObserve.insert("sycl.pi");
    GStreamsToObserve.insert("sycl.perf");
    GStreamsToObserve.insert("sycl.experimental.level_zero.call");
    GStreamsToObserve.insert("sycl.experimental.cuda.call");

    GProcessID = xpti::utils::get_process_id();
    InitStreams = false;
    std::cout << "Set up streams to observe\n";
    GWriter = new xpti::table_writer();
  };

  auto Check = GStreamsToObserve.find(stream_name);

  if (std::string("sycl") == stream_name && Check != GStreamsToObserve.end()) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::wait_begin,
                         traceCallback);
    xptiRegisterCallback(StreamID, (uint16_t)xpti::trace_point_type_t::wait_end,
                         traceCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::task_begin,
                         traceCallback);
    xptiRegisterCallback(StreamID, (uint16_t)xpti::trace_point_type_t::task_end,
                         traceCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::barrier_begin,
                         traceCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::barrier_end,
                         traceCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::graph_create,
                         graphCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::node_create,
                         graphCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::edge_create,
                         graphCallback);
  } else if (std::string("sycl.experimental.mem.alloc") == stream_name &&
             Check != GStreamsToObserve.end()) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::node_create,
                         graphMemCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::edge_create,
                         graphMemCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::mem_alloc_begin,
                         syclMemCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::mem_alloc_end,
                         syclMemCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::mem_release_begin,
                         syclMemCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::mem_release_end,
                         syclMemCallback);
  } else if (std::string("sycl.pi") == stream_name &&
             Check != GStreamsToObserve.end()) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclPiCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclPiCallback);
  } else if (std::string("sycl.perf") == stream_name &&
             Check != GStreamsToObserve.end()) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclPerfCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclPerfCallback);
  } else if (std::string("sycl.experimental.level_zero.call") == stream_name &&
             Check != GStreamsToObserve.end()) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclL0Callback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclL0Callback);
  } else if (std::string("sycl.experimental.cuda.call") == stream_name &&
             Check != GStreamsToObserve.end()) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclCudaCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclCudaCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) {
  std::lock_guard<std::mutex> _{GStreamMutex};
  // auto loc = GStreamsToObserve.find(stream_name);
  // GStreamsToObserve.erase(stream_name);
  // std::cout << "Removed stream: " << stream_name << std::endl;
  // std::cout << "Stream count: " << GStreamsToObserve.size() << std::endl;
  // if (GStreamsToObserve.empty()) {
  if (GWriter) {
    GWriter->fini();
    delete GWriter;
    GWriter = nullptr;
  }
  // }
}

void record_and_save(const char *StreamName, xpti::trace_event_data_t *Event,
                     uint16_t TraceType, uint64_t Instance,
                     const void *UserData) {
  // For `function_begin` trace point type, the Parent and Event parameters can
  // be null. However, the `UserData` field must be present and contain the
  // function name that these trace points are defined to trace.
  const char *Name = nullptr;
  if (UserData) {
    Name = (const char *)UserData;
  } else {
    Name = "unknown";
  }
  xpti::record_t r;
  uint64_t ID = 0;

  if (TraceType & 0x0001) {
    // We are the closing scope step
    {
      std::lock_guard<std::mutex> _{GRecMutex};
      uid_lookup_t &EventRecords = GIncompleteRecords[StreamName];
      if (Event)
        ID = Event->unique_id;
      else
        ID = 0;
      incomplete_records_t &IncompleteRecords = EventRecords[ID];
      auto ele = IncompleteRecords.find(Instance);
      if (ele != IncompleteRecords.end()) {
        // Copy so the incomplete record can be deleted
        r = ele->second;
        IncompleteRecords.erase(ele);
      } else {
        throw std::runtime_error("Instance id/correlation ID collision!");
      }
    }
    // We are operating on a copy, so no data races
    record_state(r, false);
    // Writer has its own lock
    if (GWriter)
      GWriter->write(r);
  } else {
    // Create record and save as we are at the begin scope step
    record_state(r, true);
    r.Name = Name;
    r.Category = StreamName;
    r.Flags |= (uint64_t)(xpti::RecordFlags::NamePresent);
    {
      std::lock_guard<std::mutex> _{GRecMutex};
      r.CorrID = Instance;
      uid_lookup_t &EventRecords = GIncompleteRecords[StreamName];
      if (Event)
        ID = Event->unique_id;
      else
        ID = 0;
      incomplete_records_t &IncompleteRecords = EventRecords[ID];
      if (IncompleteRecords.count(Instance)) {
        throw std::runtime_error("Instance id/correlation ID collision!");
      }
      IncompleteRecords[Instance] = r;
    }
  }
}

XPTI_CALLBACK_API void traceCallback(uint16_t TraceType,
                                     xpti::trace_event_data_t *Parent,
                                     xpti::trace_event_data_t *Event,
                                     uint64_t Instance, const void *UserData) {
  record_and_save("sycl", (Event ? Event : Parent), TraceType, Instance,
                  UserData);
}

XPTI_CALLBACK_API
void graphCallback(uint16_t TraceType, xpti::trace_event_data_t *Parent,
                   xpti::trace_event_data_t *Event, uint64_t Instance,
                   const void *UserData) {
  // Need to add DOT writer here
}

XPTI_CALLBACK_API void syclMemCallback(uint16_t TraceType,
                                       xpti::trace_event_data_t *Parent,
                                       xpti::trace_event_data_t *Event,
                                       uint64_t Instance,
                                       const void *UserData) {
  record_and_save("sycl.experimental.mem.alloc", (Event ? Event : Parent),
                  TraceType, Instance, UserData);
}

XPTI_CALLBACK_API void syclL0Callback(uint16_t TraceType,
                                      xpti::trace_event_data_t *Parent,
                                      xpti::trace_event_data_t *Event,
                                      uint64_t Instance, const void *UserData) {
  record_and_save("sycl.experimental.level_zero.call", (Event ? Event : Parent),
                  TraceType, Instance, UserData);
}

XPTI_CALLBACK_API void syclCudaCallback(uint16_t TraceType,
                                        xpti::trace_event_data_t *Parent,
                                        xpti::trace_event_data_t *Event,
                                        uint64_t Instance,
                                        const void *UserData) {
  record_and_save("sycl.experimental.cuda.call", (Event ? Event : Parent),
                  TraceType, Instance, UserData);
}

XPTI_CALLBACK_API void graphMemCallback(uint16_t TraceType,
                                        xpti::trace_event_data_t *Parent,
                                        xpti::trace_event_data_t *Event,
                                        uint64_t Instance,
                                        const void *UserData) {
  // Need to add DOT writer here
}

XPTI_CALLBACK_API void syclPiCallback(uint16_t TraceType,
                                      xpti::trace_event_data_t *Parent,
                                      xpti::trace_event_data_t *Event,
                                      uint64_t Instance, const void *UserData) {
  record_and_save("sycl.pi", (Event ? Event : Parent), TraceType, Instance,
                  UserData);
}

XPTI_CALLBACK_API void syclPerfCallback(uint16_t TraceType,
                                        xpti::trace_event_data_t *Parent,
                                        xpti::trace_event_data_t *Event,
                                        uint64_t Instance,
                                        const void *UserData) {
  record_and_save("sycl.perf", (Event ? Event : Parent), TraceType, Instance,
                  UserData);
}

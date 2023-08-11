//
// An example collector/tool that prints out just the SYCL PI layer trace
// events
//
#include "xpti/xpti_trace_framework.h"
#include "xpti_helpers.hpp"
#include "xpti_timers.hpp"
#include "xpti_writers.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

std::set<std::string> GStreamsToObserve;
xpti::writer *GWriter = nullptr;

uint64_t GProcessID = 0;

using correlation_records_t = std::unordered_map<uint64_t, xpti::record_t>;
correlation_records_t GIncompleteRecords;
std::mutex GRecMutex, GStreamMutex;
xpti::utils::statistics_t GEventStats;
uint64_t GOutputFormats = 0;
xpti::utils::string::list_t *GStreams = nullptr;
extern bool xpti::ShowInColors;

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
    std::set<std::string> OutputFormats{"json", "csv", "table", "stack", "all"};
    xpti::utils::string::simple_string_decoder_t D(",");
    GStreams = new xpti::utils::string::list_t;
    // Environment variables that are used to communicate runtime
    // characteristics that can be encapsulated in a launcher application
    //
    // 1. XPTI_SYCL_PERF_OUTPUT=[json,csv,table,stack,all]
    // 2. XPTI_STREAMS=[all] or [sycl,sycl.pi,sycl.perf,sycl.perf.detail,...]
    // 3. XPTI_STDOUT_USE_COLOR=[1,0]
    //
    const char *ProfOutFile = std::getenv("XPTI_SYCL_PERF_OUTPUT");
    if (ProfOutFile)
      std::cout << "XPTI_SYCL_PERF_OUTPUT=" << ProfOutFile << "\n";
    const char *StreamsToMonitor = std::getenv("XPTI_STREAMS");
    if (StreamsToMonitor)
      std::cout << "XPTI_STREAMS=" << StreamsToMonitor << "\n";
    const char *ColorOutput = std::getenv("XPTI_STDOUT_USE_COLOR");
    if (ColorOutput)
      std::cout << "XPTI_STDOUT_USE_COLOR=" << ColorOutput << "\n";

    // Get the streams to monitor from the environment variable
    // In order to determine if the environment variable contains valid stream
    // names, a catalog of all streams must be check against to validate the
    // strings
    if (StreamsToMonitor) {
      auto streams = D.decode(StreamsToMonitor);
      for (auto &s : streams) {
        GStreams->add(s.c_str());
      }
    } else {
      // If the environment variable is not set, pick the default streams we
      // would like to monitor
      GStreams->add("sycl");
      GStreams->add("sycl.experimental.mem.alloc");
      GStreams->add("sycl.pi");
      GStreams->add("sycl.perf");
      GStreams->add("sycl.experimental.level_zero.call");
      GStreams->add("sycl.experimental.cuda.call");
    }

    // Check the output format environmental variable and only accept valid
    // types
    if (ProfOutFile) {
      auto outputs = D.decode(ProfOutFile);
      for (auto &o : outputs) {
        if (OutputFormats.count(o)) {
          if (o == "json") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::JSON;
          } else if (o == "csv") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::CSV;
          } else if (o == "table") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::Table;
          } else if (o == "stack") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::Stack;
          } else if (o == "all") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::All;
          } else {
            std::cerr << "Invalid format provided: " << o
                      << " - Ignoring format!\n";
          }
        }
      }
    } else {
      GOutputFormats = (uint64_t)xpti::FileFormat::Stack;
    }

    // Check the environment variable for colored output and set the appropriate
    // flag, if the variable is set
    if (ColorOutput) {
      if (std::stoi(ColorOutput) == 0)
        xpti::ShowInColors = false;
      else
        xpti::ShowInColors = true;
    } else {
      xpti::ShowInColors = true;
    }

    GStreamsToObserve.insert("sycl");
    GStreamsToObserve.insert("sycl.experimental.mem.alloc");
    GStreamsToObserve.insert("sycl.pi");
    GStreamsToObserve.insert("sycl.perf");
    GStreamsToObserve.insert("sycl.experimental.level_zero.call");
    GStreamsToObserve.insert("sycl.experimental.cuda.call");

    // GDataModel = new xpti::data_model_t();
    GProcessID = xpti::utils::get_process_id();
    InitStreams = false;
    if (GOutputFormats & (uint64_t)xpti::FileFormat::JSON)
      GWriter = new xpti::json_writer();
    else
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

std::once_flag GFinalize;

XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) {
  std::lock_guard<std::mutex> _{GStreamMutex};
  GStreams->remove(stream_name);
  if (GStreams->empty()) {
    std::cout << "All subscribed streams have been unregistered!\n";
  }
  // auto loc = GStreamsToObserve.find(stream_name);
  // GStreamsToObserve.erase(stream_name);
  // std::cout << "Removed stream: " << stream_name << std::endl;
  // std::cout << "Stream count: " << GStreamsToObserve.size() << std::endl;
  // if (GStreamsToObserve.empty()) {
  std::call_once(GFinalize, []() {
    if (MeasureEventCost) {
      std::cout << "Event handler cost: " << std::fixed << GEventStats.mean()
                << " ns/event [" << std::setprecision(0) << GEventStats.count()
                << " events] [Min time: " << GEventStats.min() << " ns]\n";
    }
    if (GWriter) {
      GWriter->fini();
      delete GWriter;
      GWriter = nullptr;
    }
  });
  // }
}

void record_and_save(const char *StreamName, xpti::trace_event_data_t *Event,
                     uint16_t TraceType, uint64_t Instance,
                     const void *UserData) {
  // For `function_begin` trace point type, the Parent and Event parameters
  // can be null. However, the `UserData` field must be present and contain
  // the function name that these trace points are defined to trace.
  xpti::utils::timer::measurement_t m;
  uint64_t begin, end;

  if (MeasureEventCost) {
    begin = m.clock();
  }

  const char *Name = (UserData ? (const char *)UserData : "unknown");
  xpti::record_t r;

  if (TraceType & 0x0001) {
    // We are the closing scope step
    {
      std::lock_guard<std::mutex> _{GRecMutex};
      auto ele = GIncompleteRecords.find(Instance);
      if (ele != GIncompleteRecords.end()) {
        // Copy so the incomplete record can be deleted
        r = ele->second;
        GIncompleteRecords.erase(ele);
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
      if (GIncompleteRecords.count(Instance)) {
        throw std::runtime_error("Instance id/correlation ID collision!");
      }
      GIncompleteRecords[Instance] = r;
    }
  }
  if (MeasureEventCost) {
    end = m.clock();
    GEventStats.add_value(end - begin + 1);
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

//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM
//  Exceptions. See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include "xpti/xpti_trace_framework.h"

#include "xpti_helpers.hpp"
#include "xpti_timers.hpp"
#include "xpti_writers.hpp"

#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

std::mutex GRecMutex, GStreamMutex;
xpti::data_model *GDataModel = nullptr;
xpti::utils::statistics_t GEventStats;
uint64_t GProcessID = 0;
bool CalibrationRun = false;
bool MeasureEventCost = true;
bool ShowDebugInformation = false;
bool ShowVerboseOutput = false;
bool ShowInColors;

uint64_t GOutputFormats = 0;
// Data structure to keep track of the streams to monitor
xpti::utils::string::list_t *GStreams = nullptr;
xpti::utils::string::list_t GAllStreams;
// Data structure to ha
xpti::utils::string::first_check_map_t *GIgnoreList = nullptr;

using incomplete_records_t = std::unordered_map<uint64_t, xpti::record_t>;
incomplete_records_t *GRecordsInProgress = nullptr;
xpti::utils::timer::measurement_t GMeasure;

constexpr const char *GStreamBasic = "sycl";
constexpr const char *GStreamPI = "sycl.pi";
constexpr const char *GStreamMemory = "sycl.experimental.mem_alloc";
constexpr const char *GStreamL0 = "sycl.experimental.level_zero.call";
constexpr const char *GStreamCuda = "sycl.experimental.cuda.call";
constexpr const char *GStreamBuffer = "sycl.experimental.buffer";
constexpr const char *GStreamImage = "sycl.experimental.image";

// Scoped measurement object used in the callback handlers
class MeasureHandlers {
private:
  uint64_t m_begin, m_end;

public:
  MeasureHandlers() {
    if (MeasureEventCost) {
      m_begin = GMeasure.clock();
    }
  }
  ~MeasureHandlers() {
    if (MeasureEventCost) {
      m_end = GMeasure.clock();
      GEventStats.add_value(m_end - m_begin + 1);
    }
  }
};

// Implementation of the data model that will be used by all of the writers
namespace xpti {
std::mutex GDataMutex;
data_model::data_model()
    : m_min(std::numeric_limits<uint64_t>::max()), m_max(0) {}
data_model::~data_model() {}
void data_model::add(record_t &r) {
  std::lock_guard<std::mutex> _{GDataMutex};
  m_records.push_back(r);
}

void data_model::finalize() {
  std::lock_guard<std::mutex> _{GDataMutex};
  for (auto &r : m_records) {
    m_ordered_records.insert(std::make_pair(r.TSBegin, r));
    if (m_min > r.TSBegin)
      m_min = r.TSBegin;
    if (m_max < r.TSEnd)
      m_max = r.TSEnd;
  }
}

uint64_t data_model::min() { return m_min; }
uint64_t data_model::max() { return m_max; }

records_t &data_model::data() { return m_records; }
ordered_records_t &data_model::ordered_data() { return m_ordered_records; }

void data_model::print(char *Name) {
  if (ShowDebugInformation) {
    if (!Name) {
      std::cout << "++# of records: " << m_records.size() << "\n";
    } else {
      std::cout << "++# of records: " << m_records.size() << " : " << Name
                << "\n";
    }
    std::lock_guard<std::mutex> _{GDataMutex};
    for (auto &r : m_records) {
      std::cout << "--->" << r.TSBegin << " : " << r.Name << "\n";
    }
  }
}

} // namespace xpti

// Collector implementation: Implements a generic collector to present data in
// JSON, CSV and STACK formats.The primary use of this collector is to have an
// infrastructure that can measure the the cost of event generation and event
// handling.
//
// 1. Run application with XPTI_TRACE_ENABLE=0
//    This will provide the timing information where the impact of XPTI
//    instrumentation should be non-existent
// 2. Run with XPTI_TRACE_ENABLE=1
//    And enable the --calibrate option.
//    Any timing difference will be the overheads due to the XPTI infrastructure
//    and SYCL instrumentation overheads
// 3. Run it without the calibrate option and use "table" format
//    Should include overheads due to infrastructure, SYCL instrumentation and
//    callback handlers used by the collector

static void record_state(xpti::record_t &r, bool begin_scope) {
  if (begin_scope) {
    r.TSBegin = GMeasure.clock();
    r.Flags |= (uint64_t)(xpti::RecordFlags::BeginTimePresent);
    r.HWID = GMeasure.cpu();
    r.TID = GMeasure.thread();
    r.PID = GProcessID;
    r.Flags |= ((uint64_t)xpti::RecordFlags::TimeInNanoseconds);
  } else {
    r.TSEnd = GMeasure.clock();
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
XPTI_CALLBACK_API void syclMemCallback(uint16_t trace_type,
                                       xpti::trace_event_data_t *parent,
                                       xpti::trace_event_data_t *event,
                                       uint64_t instance,
                                       const void *user_data);
XPTI_CALLBACK_API void syclBufferCallback(uint16_t trace_type,
                                          xpti::trace_event_data_t *parent,
                                          xpti::trace_event_data_t *event,
                                          uint64_t instance,
                                          const void *user_data);
XPTI_CALLBACK_API void syclImageCallback(uint16_t trace_type,
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

// This is one of the functions expected to be implemented by XPTI to enable the
// data collection. The initialization function allows usrs to selectively
// capture data from various data streams

XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version,
                                     const char *version_str,
                                     const char *stream_name) {
  // On first use, set up some data structures to track the streams we are
  // registering with and create some writer objects for formatted output
  static bool InitStreams = true;
  if (InitStreams) {
    GRecordsInProgress = new incomplete_records_t;
    std::set<std::string> OutputFormats{"json", "csv", "table", "stack", "all"};
    xpti::utils::string::simple_string_decoder_t D(",");
    GStreams = new xpti::utils::string::list_t;
    if (!GStreams) {
      std::cerr
          << "Unable to allocate memory for Streams to monitor! Aborting..\n";
      exit(-1);
    }
    GIgnoreList = new xpti::utils::string::first_check_map_t;
    if (!GIgnoreList) {
      std::cerr << "Unable to allocate memory for Ignore List! Aborting..\n";
      exit(-1);
    }

    // Environment variables that are used to communicate runtime
    // characteristics that can be encapsulated in a launcher application
    //
    // 1. XPTI_SYCL_PERF_OUTPUT=[json,csv,table,stack,all]
    // 2. XPTI_STREAMS=[all] or [sycl,sycl.pi,sycl.perf,sycl.perf.detail,...]
    // 3. XPTI_STDOUT_USE_COLOR=[1,0]
    // 4. XPTI_IGNORE_LIST=piPlatformsGet,piProgramBuild
    // 5. XPTI_SIMULATION=10,20,50,100
    //

    // Set up the Verbose output flag first so we can provide progress output of
    // the collector
    const char *VerboseFlag = std::getenv("XPTI_VERBOSE");
    // Check the environment variable for verbose output and set the appropriate
    // flag, if the variable is set
    if (VerboseFlag && std::stoi(VerboseFlag) != 0)
      ShowVerboseOutput = true;
    else
      ShowVerboseOutput = false;

    const char *DebugFlag = std::getenv("XPTI_DEBUG");
    // Check the environment variable for debug output and set the appropriate
    // flag, if the variable is set
    if (DebugFlag && std::stoi(DebugFlag) != 0)
      ShowDebugInformation = true;
    else
      ShowDebugInformation = false;

    const char *ProfOutFile = std::getenv("XPTI_SYCL_PERF_OUTPUT");
    if (ShowVerboseOutput && ProfOutFile)
      std::cout << "XPTI_SYCL_PERF_OUTPUT=" << ProfOutFile << "\n";
    const char *StreamsToMonitor = std::getenv("XPTI_STREAMS");
    if (ShowVerboseOutput && StreamsToMonitor)
      std::cout << "XPTI_STREAMS=" << StreamsToMonitor << "\n";
    const char *ColorOutput = std::getenv("XPTI_STDOUT_USE_COLOR");
    if (ShowVerboseOutput && ColorOutput)
      std::cout << "XPTI_STDOUT_USE_COLOR=" << ColorOutput << "\n";
    const char *FirstCallsToIgnore = std::getenv("XPTI_IGNORE_LIST");
    if (ShowVerboseOutput && FirstCallsToIgnore)
      std::cout << "XPTI_IGNORE_LIST=" << FirstCallsToIgnore << "\n";
    const char *SimulationOverheads = std::getenv("XPTI_SIMULATION");
    if (ShowVerboseOutput && SimulationOverheads)
      std::cout << "XPTI_SIMULATION=" << SimulationOverheads << "\n";
    const char *CalibrationFlag = std::getenv("XPTI_CALIBRATE");
    if (ShowVerboseOutput && CalibrationFlag)
      std::cout << "XPTI_SIMULATION=" << CalibrationFlag << "\n";
    if (ShowVerboseOutput && VerboseFlag)
      std::cout << "XPTI_VERBOSE=" << VerboseFlag << "\n";
    if (ShowVerboseOutput && DebugFlag)
      std::cout << "XPTI_DEBUG=" << DebugFlag << "\n";

    // Check the environment variable for colored output and set the appropriate
    // flag, if the variable is set
    if (CalibrationFlag && std::stoi(CalibrationFlag) != 0)
      CalibrationRun = true;
    else
      CalibrationRun = false;

    // Get the streams to monitor from the environment variable
    // In order to determine if the environment variable contains valid stream
    // names, a catalog of all streams must be check against to validate the
    // strings
    if (ShowVerboseOutput && StreamsToMonitor) {
      std::cout << "Monitoring streams: ";

      auto streams = D.decode(StreamsToMonitor);
      for (auto &s : streams) {
        std::cout << s << ", ";
        GAllStreams.add(s.c_str());
      }
      if (ShowVerboseOutput)
        std::cout << "null\n";
    } else {
      if (ShowVerboseOutput)
        std::cout << "Streams to monitor:  default\n";
      // If the environment variable is not set, pick the default streams we
      // would like to monitor
      GAllStreams.add(GStreamBasic);
      GAllStreams.add(GStreamPI);
      GAllStreams.add(GStreamMemory);
      // GAllStreams.add(GStreamBuffer);
      GAllStreams.add(GStreamImage);
    }

    // Capture the user input on the first calls to ignore; some calls,
    // especially in SYCL BEs, pay a first time penalty and this allows the
    // statistics computation to ignore such calls
    if (FirstCallsToIgnore) {
      auto streams = D.decode(FirstCallsToIgnore);
      for (auto &s : streams) {
        GIgnoreList->add(s.c_str());
      }
    } else {
      // If the environment variable is not set, pick the default calls we
      // would like to ignore
      GIgnoreList->add("piProgramBuild");
      GIgnoreList->add("piPlatformsGet");
    }

    // Check the output format environmental variable and only accept valid
    // types
    if (ProfOutFile) {
      if (ShowVerboseOutput)
        std::cout << "Output format: ";
      auto outputs = D.decode(ProfOutFile);
      for (auto &o : outputs) {
        if (OutputFormats.count(o)) {
          if (o == "json") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::JSON;
            if (ShowVerboseOutput)
              std::cout << "JSON,";
          } else if (o == "table") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::Table;
            if (ShowVerboseOutput)
              std::cout << "Table,";
          } else if (o == "stack") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::Stack;
            if (ShowVerboseOutput)
              std::cout << "Stack,";
          } else if (o == "all") {
            GOutputFormats |= (uint64_t)xpti::FileFormat::All;
            if (ShowVerboseOutput)
              std::cout << "All,";
          } else if (o == "none") {
            GOutputFormats = 0;
            if (ShowVerboseOutput)
              std::cout << "None,";
            break;
          } else {
            std::cerr << "Invalid format provided: " << o
                      << " - Ignoring format!\n";
          }
        }
      }
      if (ShowVerboseOutput)
        std::cout << "null\n";
    } else {
      GOutputFormats = (uint64_t)xpti::FileFormat::JSON;
    }

    // Check to see if the data model is uninitialized
    // if (GDataModel.get() == nullptr) {
    if (!GDataModel) {
      // Create the data model object that is thread-safe
      // GDataModel = xpti::data_model_ptr(new xpti::data_model);
      GDataModel = new xpti::data_model;
    }
    // Capture the process ID once and use it throughout for the run
    GProcessID = xpti::utils::get_process_id();

    // Check the environment variable for colored output and set the appropriate
    // flag, if the variable is set
    if (ColorOutput) {
      if (std::stoi(ColorOutput) == 0)
        ShowInColors = false;
      else
        ShowInColors = true;
    } else {
      ShowInColors = false;
    }

    // Disable initialization
    InitStreams = false;
  } // First time inititalization complete

  if (ShowVerboseOutput)
    std::cout << "Initializing stream: " << stream_name << "\n";

  // Post initialization: Once the needed data structures are created, we
  // register callbacks to the streams requsted by the end-user;
  //
  // Check == TRUE if the stream has been requested by end-user
  //
  auto Check = GAllStreams.check(stream_name);
  if (Check)
    GStreams->add(stream_name);

  if (std::string(GStreamBasic) == stream_name && Check) {
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
  } else if (std::string(GStreamMemory) == stream_name && Check) {
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
  } else if (std::string(GStreamPI) == stream_name && Check) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclPiCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclPiCallback);
  } else if (std::string(GStreamL0) == stream_name && Check) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclL0Callback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclL0Callback);
  } else if (std::string(GStreamCuda) == stream_name && Check) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_begin,
                         syclCudaCallback);
    xptiRegisterCallback(StreamID,
                         (uint16_t)xpti::trace_point_type_t::function_end,
                         syclCudaCallback);
  } else if (std::string(GStreamBuffer) == stream_name && Check) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_offload_alloc_memory_object_construct,
        syclBufferCallback);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_offload_alloc_memory_object_associate,
        syclBufferCallback);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_offload_alloc_memory_object_release,
        syclBufferCallback);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_offload_alloc_memory_object_destruct,
        syclBufferCallback);
  } else if (std::string(GStreamImage) == stream_name && Check) {
    auto StreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_offload_alloc_memory_object_construct,
        syclImageCallback);
    xptiRegisterCallback(
        StreamID, (uint16_t)xpti::trace_offload_alloc_memory_object_destruct,
        syclImageCallback);
  }
}

std::once_flag GFinalize, GCompaction;
// xptiTraceFinish is called for every stream created by the application
// software; We check to see if the 'stream' requested by the end-user has
// terminated sending events in its stream
XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) {
  {
    if (ShowVerboseOutput)
      std::cout << "Unregistering stream: " << stream_name << "\n";
    std::lock_guard<std::mutex> _{GStreamMutex};
    // Filter the streams to what is requested by the user that intersects with
    // the streams being generated by the application and middleware
    // std::call_once(GCompaction, []() { GStreams->compact(); });

    GStreams->remove(stream_name);
  }

  // When we have received notification from the infrastructure that all the
  // registered streams have closed their data streams, we finalize the data and
  // serialize to file in the format(s) requested from the command line
  if (GStreams->empty()) {
    std::call_once(GFinalize, []() {
      if (MeasureEventCost && !CalibrationRun) {
        std::cout << "Event handler cost: " << std::fixed << GEventStats.mean()
                  << " ns/event [" << std::setprecision(0)
                  << GEventStats.count()
                  << " events] [Min time: " << GEventStats.min() << " ns]\n";
      }
      if (ShowVerboseOutput)
        std::cout << "All subscribed streams have been unregistered!\n";

      if (ShowDebugInformation)
        GDataModel->print();
      // At this point, the XPTI framework has indicated that all streams we
      // have subscribed have completed sending their events. We can use the
      // data model we have created to output in one or many formats.
      if (GDataModel && !CalibrationRun) {
        // Only create an ordered data set if an output file format is requsted
        if (GOutputFormats)
          GDataModel->finalize();

        if (GOutputFormats & (uint16_t)xpti::FileFormat::Table) {
          xpti::writer *writer =
              new xpti::table_writer(GIgnoreList, GDataModel);
          if (writer) {
            writer->fini();
            delete writer;
          }
        }
        if (GOutputFormats & (uint16_t)xpti::FileFormat::Stack) {
          xpti::writer *writer = new xpti::stack_writer(
              GIgnoreList, GDataModel, ShowDebugInformation, ShowInColors);
          if (writer) {
            writer->fini();
            delete writer;
          }
        }
        if (GOutputFormats & (uint16_t)xpti::FileFormat::JSON) {
          xpti::writer *writer = new xpti::json_writer(GDataModel);
          if (writer) {
            writer->fini();
            delete writer;
          }
        }
        delete GDataModel;
        GDataModel = nullptr;
      }
    });
  }
}

// Primarily used when debugging has been turned on
void print_record(uint16_t TraceType, xpti::record_t &r) {
  static int count = 1;
  const char *trace_origin = nullptr;

  switch (TraceType) {
  case (uint16_t)xpti::trace_point_type_t::graph_create:
    trace_origin = "graph_create";
    break;
  case (uint16_t)xpti::trace_point_type_t::node_create:
    trace_origin = "node_create";
    break;
  case (uint16_t)xpti::trace_point_type_t::edge_create:
    trace_origin = "edge_create";
    break;
  case (uint16_t)xpti::trace_point_type_t::region_begin:
    trace_origin = "region_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::region_end:
    trace_origin = "region_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::task_begin:
    trace_origin = "task_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::task_end:
    trace_origin = "task_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::barrier_begin:
    trace_origin = "barrier_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::barrier_end:
    trace_origin = "barrier_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::lock_begin:
    trace_origin = "lock_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::lock_end:
    trace_origin = "lock_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::signal:
    trace_origin = "signal";
    break;
  case (uint16_t)xpti::trace_point_type_t::transfer_begin:
    trace_origin = "transfer_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::transfer_end:
    trace_origin = "transfer_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::thread_begin:
    trace_origin = "thread_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::thread_end:
    trace_origin = "thread_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::wait_begin:
    trace_origin = "wait_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::wait_end:
    trace_origin = "wait_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::function_begin:
    trace_origin = "function_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::function_end:
    trace_origin = "function_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::metadata:
    trace_origin = "metadata";
    break;
  case (uint16_t)xpti::trace_point_type_t::function_with_args_begin:
    trace_origin = "function_with_args_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::function_with_args_end:
    trace_origin = "function_with_args_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::mem_alloc_begin:
    trace_origin = "mem_alloc_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::mem_alloc_end:
    trace_origin = "mem_alloc_end";
    break;
  case (uint16_t)xpti::trace_point_type_t::mem_release_begin:
    trace_origin = "mem_release_begin";
    break;
  case (uint16_t)xpti::trace_point_type_t::mem_release_end:
    trace_origin = "mem_release_end";
    break;
  case (
      uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_construct:
    trace_origin = "offload_alloc_construct";
    break;
  case (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_destruct:
    trace_origin = "offload_alloc_destruct";
    break;
  case (
      uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_associate:
    trace_origin = "offload_alloc_associate";
    break;
  case (uint16_t)xpti::trace_point_type_t::offload_alloc_memory_object_release:
    trace_origin = "offload_alloc_release";
    break;
  case (uint16_t)xpti::trace_point_type_t::offload_alloc_accessor:
    trace_origin = "offload_alloc_accessor";
    break;
  case (uint16_t)xpti::trace_point_type_t::queue_create:
    trace_origin = "queue_create";
    break;
  case (uint16_t)xpti::trace_point_type_t::queue_destroy:
    trace_origin = "queue_destroy";
    break;
  case (uint16_t)xpti::trace_point_type_t::diagnostics:
    trace_origin = "diagnostics";
    break;
  default:
    trace_origin = "Unknown";
    break;
  }

  if (ShowDebugInformation) {
    std::cout << "Record[" << count++ << "] -> " << trace_origin << " details("
              << r.TSBegin << "," << r.CorrID << "," << r.Name << ")\n";
  }
}

void record_and_save(const char *StreamName, xpti::trace_event_data_t *Event,
                     uint16_t TraceType, uint64_t Instance,
                     const void *UserData) {
  if (GRecordsInProgress) {

    // For `function_begin` trace point type, the Parent and Event parameters
    // can be null. However, the `UserData` field must be present and contain
    // the function name that these trace points are defined to trace.
    MeasureHandlers timer;

    char *Name;
    // Register string - it may have already been registered, so we will get a
    // populated string pointer that has a lifetime of XPTI framework
    xptiRegisterString((UserData ? (const char *)UserData : "unknown"), &Name);
    xpti::record_t r;

    if (TraceType & 0x0001) {
      // We are the closing scope step
      {
        std::lock_guard<std::mutex> _{GRecMutex};
        auto ele = GRecordsInProgress->find(Instance);
        if (ele != GRecordsInProgress->end()) {
          // Copy so the incomplete record can be deleted
          r = ele->second;
          GRecordsInProgress->erase(ele);
        } else {
          throw std::runtime_error("Instance id/correlation ID collision!");
        }
        // We are operating on a copy, so no data races
        record_state(r, false);
        if (ShowDebugInformation)
          print_record(TraceType, r);
        // if (GDataModel.get())
        if (GDataModel)
          GDataModel->add(r);
      }
    } else {
      // Create record and save as we are at the begin scope step
      record_state(r, true);
      r.Name = Name;
      if (GOutputFormats & (uint64_t)xpti::FileFormat::JSON)
        r.Category = StreamName;
      r.Flags |= (uint64_t)(xpti::RecordFlags::NamePresent);
      {
        std::lock_guard<std::mutex> _{GRecMutex};
        r.CorrID = Instance;
        if (GRecordsInProgress->count(Instance)) {
          throw std::runtime_error("Instance id/correlation ID collision!");
        }
        (*GRecordsInProgress)[Instance] = r;
      }
    }
  }
}

XPTI_CALLBACK_API void traceCallback(uint16_t TraceType,
                                     xpti::trace_event_data_t *Parent,
                                     xpti::trace_event_data_t *Event,
                                     uint64_t Instance, const void *UserData) {
  if (CalibrationRun)
    return;

  const char *UD = (Event->reserved.payload->name)
                       ? Event->reserved.payload->name
                       : "Unknown";
  record_and_save(GStreamBasic, (Event ? Event : Parent), TraceType, Instance,
                  UD);
}

XPTI_CALLBACK_API
void graphCallback(uint16_t TraceType, xpti::trace_event_data_t *Parent,
                   xpti::trace_event_data_t *Event, uint64_t Instance,
                   const void *UserData) {
  if (CalibrationRun)
    return;
  // Need to add DOT writer here
}

XPTI_CALLBACK_API void syclMemCallback(uint16_t TraceType,
                                       xpti::trace_event_data_t *Parent,
                                       xpti::trace_event_data_t *Event,
                                       uint64_t Instance,
                                       const void *UserData) {
  if (CalibrationRun)
    return;
  record_and_save(GStreamMemory, (Event ? Event : Parent), TraceType, Instance,
                  (TraceType & 0x0001) ? "memory_allocation_end"
                                       : "memory_allocation_begin");
}

XPTI_CALLBACK_API void syclImageCallback(uint16_t TraceType,
                                         xpti::trace_event_data_t *Parent,
                                         xpti::trace_event_data_t *Event,
                                         uint64_t Instance,
                                         const void *UserData) {
  if (CalibrationRun)
    return;
  record_and_save(GStreamImage, (Event ? Event : Parent), TraceType, Instance,
                  (TraceType & 0x0001) ? "image_destruct" : "image_construct");
}

XPTI_CALLBACK_API void syclBufferCallback(uint16_t TraceType,
                                          xpti::trace_event_data_t *Parent,
                                          xpti::trace_event_data_t *Event,
                                          uint64_t Instance,
                                          const void *UserData) {
  if (CalibrationRun)
    return;

  if (TraceType == xpti::trace_offload_alloc_memory_object_construct ||
      TraceType == xpti::trace_offload_alloc_memory_object_destruct) {
    const char *UD =
        (Event->reserved.payload->name)
            ? Event->reserved.payload->name
            : ((TraceType & 0x0001) ? "buffer_allocation_destruct"
                                    : "buffer_allocation_construct");
    record_and_save(GStreamBuffer, (Event ? Event : Parent), TraceType,
                    Instance, UD);
  } else {
    const char *UD =
        (Event->reserved.payload->name)
            ? Event->reserved.payload->name
            : ((TraceType & 0x0001) ? "buffer_allocation_release"
                                    : "buffer_allocation_associate");
    record_and_save(GStreamBuffer, (Event ? Event : Parent), TraceType,
                    Instance, UD);
  }
}

XPTI_CALLBACK_API void syclL0Callback(uint16_t TraceType,
                                      xpti::trace_event_data_t *Parent,
                                      xpti::trace_event_data_t *Event,
                                      uint64_t Instance, const void *UserData) {
  if (CalibrationRun)
    return;
  record_and_save(GStreamL0, (Event ? Event : Parent), TraceType, Instance,
                  UserData);
}

XPTI_CALLBACK_API void syclCudaCallback(uint16_t TraceType,
                                        xpti::trace_event_data_t *Parent,
                                        xpti::trace_event_data_t *Event,
                                        uint64_t Instance,
                                        const void *UserData) {
  if (CalibrationRun)
    return;
  record_and_save(GStreamCuda, (Event ? Event : Parent), TraceType, Instance,
                  UserData);
}

XPTI_CALLBACK_API void graphMemCallback(uint16_t TraceType,
                                        xpti::trace_event_data_t *Parent,
                                        xpti::trace_event_data_t *Event,
                                        uint64_t Instance,
                                        const void *UserData) {
  if (CalibrationRun)
    return;
  // Need to add DOT writer here
}

XPTI_CALLBACK_API void syclPiCallback(uint16_t TraceType,
                                      xpti::trace_event_data_t *Parent,
                                      xpti::trace_event_data_t *Event,
                                      uint64_t Instance, const void *UserData) {
  if (CalibrationRun)
    return;
  record_and_save(GStreamPI, (Event ? Event : Parent), TraceType, Instance,
                  UserData);
}

#if (defined(_WIN32) || defined(_WIN64))

#include <string>
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fwdReason, LPVOID lpvReserved) {
  switch (fwdReason) {
  case DLL_PROCESS_ATTACH:
    // printf("Framework initialization\n");
    break;
  case DLL_PROCESS_DETACH:
    //
    //  We cannot unload all subscribers here...
    //
    // printf("Framework finalization\n");
    break;
  }

  return TRUE;
}

#else // Linux (possibly macOS?)

__attribute__((constructor)) static void framework_init() {
  if (ShowVerboseOutput)
    std::cout << "Collector loaded\n";
}

__attribute__((destructor)) static void framework_fini() {
  if (GRecordsInProgress) {
    delete GRecordsInProgress;
    GRecordsInProgress = nullptr;
  }
  if (ShowVerboseOutput)
    std::cout << "Collector unloaded\n";
}

#endif

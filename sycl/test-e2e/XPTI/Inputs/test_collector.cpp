#include "xpti/xpti_trace_framework.hpp"

#include <iostream>
#include <mutex>
#include <string_view>

std::mutex GMutex;

XPTI_CALLBACK_API void syclCallback(uint16_t, xpti::trace_event_data_t *,
                                    xpti::trace_event_data_t *, uint64_t,
                                    const void *);
XPTI_CALLBACK_API void syclPiCallback(uint16_t, xpti::trace_event_data_t *,
                                      xpti::trace_event_data_t *, uint64_t,
                                      const void *);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int MajorVersion,
                                     unsigned int MinorVersion,
                                     const char *VersionStr,
                                     const char *StreamName) {
  std::cout << "xptiTraceInit: Stream Name = " << StreamName << "\n";
  std::string_view NameView{StreamName};

  if (NameView == "sycl.pi") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::function_begin),
        syclPiCallback);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end),
        syclPiCallback);
  }
  if (NameView == "sycl") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::graph_create),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::node_create),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::edge_create),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::task_begin),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::task_end),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::signal),
        syclCallback);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::barrier_begin),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::barrier_end),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::wait_begin),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::wait_end),
        syclCallback);
    xptiRegisterCallback(
        StreamID, static_cast<uint16_t>(xpti::trace_point_type_t::signal),
        syclCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *streamName) {
  std::cout << "xptiTraceFinish: Stream Name = " << streamName << "\n";
}

XPTI_CALLBACK_API void syclPiCallback(uint16_t TraceType,
                                      xpti::trace_event_data_t *,
                                      xpti::trace_event_data_t *, uint64_t,
                                      const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  const char *funcName = static_cast<const char *>(UserData);
  if (Type == xpti::trace_point_type_t::function_begin) {
    std::cout << "PI Call Begin : ";
  } else if (Type == xpti::trace_point_type_t::function_end) {
    std::cout << "PI Call End : ";
  }
  std::cout << funcName << "\n";
}

XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *,
                                    xpti::trace_event_data_t *Event, uint64_t,
                                    const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  switch (Type) {
  case xpti::trace_point_type_t::graph_create:
    std::cout << "Graph create\n";
    break;
  case xpti::trace_point_type_t::node_create:
    std::cout << "Node create\n";
    break;
  case xpti::trace_point_type_t::edge_create:
    std::cout << "Edge create\n";
    break;
  case xpti::trace_point_type_t::task_begin:
    std::cout << "Task begin\n";
    break;
  case xpti::trace_point_type_t::task_end:
    std::cout << "Task end\n";
    break;
  case xpti::trace_point_type_t::signal:
    std::cout << "Signal\n";
    break;
  case xpti::trace_point_type_t::wait_begin:
    std::cout << "Wait begin\n";
    break;
  case xpti::trace_point_type_t::wait_end:
    std::cout << "Wait end\n";
    break;
  case xpti::trace_point_type_t::barrier_begin:
    std::cout << "Barrier begin\n";
    break;
  case xpti::trace_point_type_t::barrier_end:
    std::cout << "Barrier end\n";
    break;
  default:
    std::cout << "Unknown tracepoint\n";
  }

  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (auto &Item : *Metadata) {
    std::cout << "  " << xptiLookupString(Item.first) << " : "
              << xptiLookupString(Item.second) << "\n";
  }
}

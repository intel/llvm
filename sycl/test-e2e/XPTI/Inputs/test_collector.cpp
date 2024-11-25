#include "xpti/xpti_trace_framework.hpp"

#include <iostream>
#include <mutex>
#include <string_view>

std::mutex GMutex;

XPTI_CALLBACK_API void syclCallback(uint16_t, xpti::trace_event_data_t *,
                                    xpti::trace_event_data_t *, uint64_t,
                                    const void *);
XPTI_CALLBACK_API void syclUrCallback(uint16_t, xpti::trace_event_data_t *,
                                      xpti::trace_event_data_t *, uint64_t,
                                      const void *);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int MajorVersion,
                                     unsigned int MinorVersion,
                                     const char *VersionStr,
                                     const char *StreamName) {
  std::cout << "xptiTraceInit: Stream Name = " << StreamName << "\n";
  std::string_view NameView{StreamName};
  using type = xpti::trace_point_type_t;

  if (NameView == "ur.call") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    for (type t : std::initializer_list<type>{type::function_with_args_begin})
      xptiRegisterCallback(StreamID, static_cast<uint16_t>(t), syclUrCallback);
  }
  if (NameView == "sycl") {
    uint8_t StreamID = xptiRegisterStream(StreamName);

    for (type t : std::initializer_list<type>{
             type::graph_create, type::node_create, type::edge_create,
             type::task_begin, type::task_end, type::signal,
             type::barrier_begin, type::barrier_end, type::wait_begin,
             type::wait_end})
      xptiRegisterCallback(StreamID, static_cast<uint16_t>(t), syclCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *streamName) {
  std::cout << "xptiTraceFinish: Stream Name = " << streamName << "\n";
}

XPTI_CALLBACK_API void syclUrCallback(uint16_t TraceType,
                                      xpti::trace_event_data_t *,
                                      xpti::trace_event_data_t *, uint64_t,
                                      const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  auto *args = static_cast<const xpti::function_with_args_t *>(UserData);
  if (Type == xpti::trace_point_type_t::function_with_args_begin) {
    std::cout << "UR Call Begin : ";
  } else if (Type == xpti::trace_point_type_t::function_with_args_end) {
    std::cout << "UR Call End : ";
  }
  std::cout << args->function_name << "\n";
}

XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *,
                                    xpti::trace_event_data_t *Event, uint64_t,
                                    const void *UserData) {
  char *Key = 0;
  uint64_t Value;
  bool HaveKeyValue =
      (xptiGetStashedTuple(&Key, Value) == xpti::result_t::XPTI_RESULT_SUCCESS);
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

  if (HaveKeyValue) {
    std::cout << "  " << Key << " : " << Value << "\n";
  }
  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (auto &Item : *Metadata) {
    std::cout << "  " << xptiLookupString(Item.first) << " : "
              << xpti::readMetadata(Item) << "\n";
  }
}

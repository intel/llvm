#include "xpti/xpti_trace_framework.hpp"

#include <iostream>
#include <mutex>
#include <string_view>

std::mutex GMutex;

XPTI_CALLBACK_API void memCallback(uint16_t, xpti::trace_event_data_t *,
                                   xpti::trace_event_data_t *, uint64_t,
                                   const void *);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int MajorVersion,
                                     unsigned int MinorVersion,
                                     const char *VersionStr,
                                     const char *StreamName) {
  std::cout << "xptiTraceInit: Stream Name = " << StreamName << "\n";
  std::string_view NameView{StreamName};

  if (NameView == "sycl.experimental.mem_alloc") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::mem_alloc_begin),
        memCallback);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::mem_alloc_end),
        memCallback);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::mem_release_begin),
        memCallback);
    xptiRegisterCallback(
        StreamID,
        static_cast<uint16_t>(xpti::trace_point_type_t::mem_release_end),
        memCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *streamName) {
  std::cout << "xptiTraceFinish: Stream Name = " << streamName << "\n";
}

XPTI_CALLBACK_API void memCallback(uint16_t TraceType,
                                   xpti::trace_event_data_t *,
                                   xpti::trace_event_data_t *, uint64_t,
                                   const void *UserData) {
  std::lock_guard Lock{GMutex};
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  auto *Data = static_cast<const xpti::mem_alloc_data_t *>(UserData);
  if (Type == xpti::trace_point_type_t::mem_alloc_begin) {
    std::cout << "Mem Alloc Begin : ";
  } else if (Type == xpti::trace_point_type_t::mem_alloc_end) {
    std::cout << "Mem Alloc End : ";
  } else if (Type == xpti::trace_point_type_t::mem_release_begin) {
    std::cout << "Mem Release Begin : ";
  } else if (Type == xpti::trace_point_type_t::mem_release_end) {
    std::cout << "Mem Release End : ";
  }
  std::cout << "  mem_obj_handle: " << Data->mem_object_handle << "\n";
  std::cout << "  alloc_pointer : " << Data->alloc_pointer << "\n";
  std::cout << "  alloc_size    : " << Data->alloc_size << "\n";
}

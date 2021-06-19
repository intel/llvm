#include "xpti_trace_framework.h"

#include "pi_arguments_handler.hpp"

#include <detail/plugin_printers.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

static uint8_t GStreamID = 0;
std::mutex GIOMutex;

sycl::xpti_helpers::PiArgumentsHandler ArgHandler;

// The lone callback function we are going to use to demonstrate how to attach
// the collector to the running executable
XPTI_CALLBACK_API void tpCallback(uint16_t trace_type,
                                  xpti::trace_event_data_t *parent,
                                  xpti::trace_event_data_t *event,
                                  uint64_t instance, const void *user_data);

// Based on the documentation, every subscriber MUST implement the
// xptiTraceInit() and xptiTraceFinish() APIs for their subscriber collector to
// be loaded successfully.
XPTI_CALLBACK_API void xptiTraceInit(unsigned int major_version,
                                     unsigned int minor_version,
                                     const char *version_str,
                                     const char *stream_name) {
  if (std::string_view(stream_name) == "sycl.pi.arg") {
    GStreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(
        GStreamID, (uint16_t)xpti::trace_point_type_t::function_with_args_begin,
        tpCallback);
    xptiRegisterCallback(
        GStreamID, (uint16_t)xpti::trace_point_type_t::function_with_args_end,
        tpCallback);

#define _PI_API(api)                                                           \
  ArgHandler.set##_##api([](auto &&...Args) {                                  \
    std::cout << "--->" << #api << "("                                         \
              << "\n";                                                         \
    sycl::detail::pi::printArgs(Args...);                                      \
    std::cout << ")\n";                                                        \
  });
#include <CL/sycl/detail/pi.def>
#undef _PI_API
    ArgHandler.set_piProgramBuild([](auto &&...) {});
    ArgHandler.set_piEnqueueMemBufferRead([](auto &&...) {});
    ArgHandler.set_piEventsWait([](auto &&...) {});
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *stream_name) {
  // NOP
}

XPTI_CALLBACK_API void tpCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData) {
  auto Type = static_cast<xpti::trace_point_type_t>(TraceType);
  if (Type == xpti::trace_point_type_t::function_with_args_begin) {
    // Lock while we print information
    std::lock_guard<std::mutex> Lock(GIOMutex);

    const auto *Data =
        static_cast<const xpti::function_with_args_t *>(UserData);

    ArgHandler.handle(Data->function_id, Data->args_data);
  }
}

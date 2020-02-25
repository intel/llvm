//==---------------- pi_hooks.cpp - SYCL standard source file --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti_trace_framework.hpp"
#endif // XPTI_ENABLE_INSTRUMENTATION

#include <detail/config.hpp>
#include <pi/pi.hpp>

namespace pi {

namespace config {

TraceLevel trace_level_mask() {
  using namespace cl::sycl::detail;
  return static_cast<TraceLevel>(SYCLConfig<SYCL_PI_TRACE>::get());
}

pi::backend *backend() {
  using namespace cl::sycl::detail;
  return SYCLConfig<SYCL_BE>::get();
}

pi::device_filter_list *device_filter_list() {
  using namespace cl::sycl::detail;
  return SYCLConfig<SYCL_DEVICE_FILTER>::get();
}

} // namespace config

#ifdef XPTI_ENABLE_INSTRUMENTATION

xpti::trace_event_data_t *GSYCLGraphEvent = nullptr;

const char *SYCL_STREAM_NAME = "sycl";
#endif // XPTI_ENABLE_INSTRUMENTATION

} // namespace pi

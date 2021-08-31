//==---------- xpti_registry.hpp ----- XPTI Stream Registry ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <unordered_set>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// We define a sycl stream name and this will be used by the instrumentation
// framework
inline constexpr const char *SYCL_STREAM_NAME = "sycl";
// Stream name being used for traces generated from the SYCL plugin layer
inline constexpr const char *SYCL_PICALL_STREAM_NAME = "sycl.pi";
// Stream name being used for traces generated from PI calls. This stream
// contains information about function arguments.
inline constexpr const char *SYCL_PIDEBUGCALL_STREAM_NAME = "sycl.pi.debug";

class XPTIRegistry {
public:
  /// Notifies XPTI subscribers about new stream.
  ///
  /// \param StreamName is a name of newly initialized stream.
  /// \param MajVer is a stream major version.
  /// \param MinVer is a stream minor version.
  /// \param VerStr is a string of "MajVer.MinVer" format.
  void initializeStream(const std::string &StreamName, uint32_t MajVer,
                        uint32_t MinVer, const std::string &VerStr) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    MActiveStreams.insert(StreamName);
    xptiInitialize(StreamName.c_str(), MajVer, MinVer, VerStr.c_str());
#endif // XPTI_ENABLE_INSTRUMENTATION
  }

  ~XPTIRegistry() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    for (const auto &StreamName : MActiveStreams) {
      xptiFinalize(StreamName.c_str());
    }
#endif // XPTI_ENABLE_INSTRUMENTATION
  }

private:
  std::unordered_set<std::string> MActiveStreams;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

//==---------- plugin_pi_hooks.cpp - PI library hooks ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pi/pi.hpp>

// We just need basic definitions for the plugins to link properly
// These symbols aren't actually used in plugins,
// any library linking against piapi directly
// still needs to define its own hooks

namespace pi {
namespace config {

TraceLevel trace_level_mask() { return {TraceLevel::PI_TRACE_ALL}; }
pi::backend *backend() { return nullptr; }
pi::device_filter_list *device_filter_list() { return nullptr; }

} // namespace config
} // namespace pi

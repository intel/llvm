//===--------- event.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "event.hpp"
#include "event_provider.hpp"
#include "ze_api.h"

namespace v2 {
void ur_event::reset() {
  // consider make an abstraction for regular/counter based
  // events if there's more of this type of conditions
  if (type == event_type::EVENT_REGULAR) {
    zeEventHostReset(zeEvent.get());
  }
}

ze_event_handle_t ur_event::getZeEvent() { return zeEvent.get(); }

ur_event::ur_event(event_allocation eventAllocation)
    : type(eventAllocation.type), zeEvent(std::move(eventAllocation.borrow)) {}

} // namespace v2

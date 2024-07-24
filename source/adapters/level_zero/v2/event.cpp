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
#include "adapters/level_zero/v2/event_provider.hpp"
#include "ze_api.h"

namespace v2 {
void ur_event::attachZeHandle(event_allocation event) {
  type = event.type;
  zeEvent = std::move(event.borrow);
}

event_borrowed ur_event::detachZeHandle() {
  // consider make an abstraction for regular/counter based
  // events if there's more of this type of conditions
  if (type == event_type::EVENT_REGULAR) {
    zeEventHostReset(zeEvent.get());
  }
  auto e = std::move(zeEvent);
  zeEvent = nullptr;

  return e;
}

ze_event_handle_t ur_event::getZeEvent() { return zeEvent.get(); }

} // namespace v2

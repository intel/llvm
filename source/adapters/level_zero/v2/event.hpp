//===--------- event.hpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <stack>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>

#include "event_provider.hpp"

namespace v2 {

class ur_event {
public:
  void attachZeHandle(event_allocation);
  event_borrowed detachZeHandle();

  ze_event_handle_t getZeEvent();

private:
  event_type type;
  event_borrowed zeEvent;
};

} // namespace v2

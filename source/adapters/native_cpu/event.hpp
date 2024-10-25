//===----------- event.hpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "common.hpp"
#include "ur_api.h"
#include <cstdint>
#include <future>
#include <mutex>
#include <vector>

struct ur_event_handle_t_ : RefCounted {

  ur_event_handle_t_(ur_queue_handle_t queue, ur_command_t command_type);

  ~ur_event_handle_t_();

  void set_callback(const std::function<void()> &cb) { callback = cb; }

  void wait();

  uint32_t getExecutionStatus() {
    // TODO: add support for UR_EVENT_STATUS_RUNNING
    std::lock_guard<std::mutex> lock(mutex);
    if (done) {
      return UR_EVENT_STATUS_COMPLETE;
    }
    return UR_EVENT_STATUS_SUBMITTED;
  }

  ur_queue_handle_t getQueue() const { return queue; }

  ur_context_handle_t getContext() const { return context; }

  ur_command_t getCommandType() const { return command_type; }

  void set_futures(std::vector<std::future<void>> &fs) {
    std::lock_guard<std::mutex> lock(mutex);
    futures = std::move(fs);
  }

  void tick_start();

  void tick_end();

  uint64_t get_start_timestamp() const { return timestamp_start; }

  uint64_t get_end_timestamp() const { return timestamp_end; }

private:
  ur_queue_handle_t queue;
  ur_context_handle_t context;
  ur_command_t command_type;
  bool done;
  std::mutex mutex;
  std::vector<std::future<void>> futures;
  std::function<void()> callback;
  uint64_t timestamp_start = 0;
  uint64_t timestamp_end = 0;
};

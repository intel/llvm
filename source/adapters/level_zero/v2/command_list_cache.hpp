//===--------- command_list_cache.hpp - Level Zero Adapter ---------------===//
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

#include "common.hpp"

namespace v2 {
namespace raii {
using ze_command_list_t = std::unique_ptr<::_ze_command_list_handle_t,
                                          decltype(&zeCommandListDestroy)>;
}

struct immediate_command_list_descriptor_t {
  ze_device_handle_t ZeDevice;
  bool IsInOrder;
  uint32_t Ordinal;
  ze_command_queue_mode_t Mode;
  ze_command_queue_priority_t Priority;
  std::optional<uint32_t> Index;
  bool operator==(const immediate_command_list_descriptor_t &rhs) const;
};

struct regular_command_list_descriptor_t {
  ze_device_handle_t ZeDevice;
  bool IsInOrder;
  uint32_t Ordinal;
  bool operator==(const regular_command_list_descriptor_t &rhs) const;
};

using command_list_descriptor_t =
    std::variant<immediate_command_list_descriptor_t,
                 regular_command_list_descriptor_t>;

struct command_list_descriptor_hash_t {
  inline size_t operator()(const command_list_descriptor_t &desc) const;
};

struct command_list_cache_t {
  command_list_cache_t(ze_context_handle_t ZeContext);

  raii::ze_command_list_t
  getImmediateCommandList(ze_device_handle_t ZeDevice, bool IsInOrder,
                          uint32_t Ordinal, ze_command_queue_mode_t Mode,
                          ze_command_queue_priority_t Priority,
                          std::optional<uint32_t> Index = std::nullopt);
  raii::ze_command_list_t getRegularCommandList(ze_device_handle_t ZeDevice,
                                                bool IsInOrder,
                                                uint32_t Ordinal);

  void addImmediateCommandList(raii::ze_command_list_t cmdList,
                               ze_device_handle_t ZeDevice, bool IsInOrder,
                               uint32_t Ordinal, ze_command_queue_mode_t Mode,
                               ze_command_queue_priority_t Priority,
                               std::optional<uint32_t> Index = std::nullopt);
  void addRegularCommandList(raii::ze_command_list_t cmdList,
                             ze_device_handle_t ZeDevice, bool IsInOrder,
                             uint32_t Ordinal);

private:
  ze_context_handle_t ZeContext;
  std::unordered_map<command_list_descriptor_t,
                     std::stack<raii::ze_command_list_t>,
                     command_list_descriptor_hash_t>
      ZeCommandListCache;
  ur_mutex ZeCommandListCacheMutex;

  raii::ze_command_list_t getCommandList(const command_list_descriptor_t &desc);
  void addCommandList(const command_list_descriptor_t &desc,
                      raii::ze_command_list_t cmdList);
  raii::ze_command_list_t
  createCommandList(const command_list_descriptor_t &desc);
};
} // namespace v2

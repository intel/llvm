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

#include <functional>
#include <stack>

#include "latency_tracker.hpp"
#include <ur/ur.hpp>
#include <ur_ddi.h>
#include <ze_api.h>

#include "common.hpp"

namespace v2 {
namespace raii {
using command_list_unique_handle =
    std::unique_ptr<::_ze_command_list_handle_t,
                    std::function<void(::ze_command_list_handle_t)>>;
} // namespace raii

struct supported_extensions_descriptor_t {
  supported_extensions_descriptor_t(bool ZeCopyOffloadExtensionSupported,
                                    bool ZeMutableCmdListExtentionSupported)
      : ZeCopyOffloadExtensionSupported(ZeCopyOffloadExtensionSupported),
        ZeMutableCmdListExtentionSupported(ZeMutableCmdListExtentionSupported) {
  }
  bool ZeCopyOffloadExtensionSupported;
  bool ZeMutableCmdListExtentionSupported;
};

struct command_list_desc_t {
  command_list_desc_t() = default;
  command_list_desc_t(bool IsInOrder, uint32_t Ordinal, bool CopyOffloadEnable,
                      bool Mutable = false)
      : IsInOrder(IsInOrder), Ordinal(Ordinal),
        CopyOffloadEnable(CopyOffloadEnable), Mutable(Mutable) {}
  bool IsInOrder;
  uint32_t Ordinal;
  bool CopyOffloadEnable;
  // The mutable command lists are not supported by immediate command lists
  // so this field will be ignored during creation of immediate command list
  bool Mutable;
};

struct immediate_command_list_descriptor_t {
  ze_device_handle_t ZeDevice;
  bool IsInOrder;
  uint32_t Ordinal;
  bool CopyOffloadEnabled;
  ze_command_queue_mode_t Mode;
  ze_command_queue_priority_t Priority;
  std::optional<uint32_t> Index;
  bool operator==(const immediate_command_list_descriptor_t &rhs) const;
};

struct regular_command_list_descriptor_t {
  ze_device_handle_t ZeDevice;
  bool IsInOrder;
  uint32_t Ordinal;
  bool CopyOffloadEnabled;
  bool Mutable;
  bool operator==(const regular_command_list_descriptor_t &rhs) const;
};

using command_list_descriptor_t =
    std::variant<immediate_command_list_descriptor_t,
                 regular_command_list_descriptor_t>;

struct command_list_descriptor_hash_t {
  inline size_t operator()(const command_list_descriptor_t &desc) const;
};

struct command_list_cache_t {
  command_list_cache_t(ze_context_handle_t ZeContext,
                       supported_extensions_descriptor_t SupportedExtensions);

  raii::command_list_unique_handle
  getImmediateCommandList(ze_device_handle_t ZeDevice, command_list_desc_t Desc,
                          ze_command_queue_mode_t Mode,
                          ze_command_queue_priority_t Priority,
                          std::optional<uint32_t> Index = std::nullopt);
  raii::command_list_unique_handle
  getRegularCommandList(ze_device_handle_t ZeDevice, command_list_desc_t Desc);

  // For testing purposes
  size_t getNumImmediateCommandLists();
  size_t getNumRegularCommandLists();

private:
  ze_context_handle_t ZeContext;
  bool ZeCopyOffloadExtensionSupported;
  bool ZeMutableCmdListExtentionSupported;
  std::unordered_map<command_list_descriptor_t,
                     std::stack<raii::ze_command_list_handle_t>,
                     command_list_descriptor_hash_t>
      ZeCommandListCache;
  ur_mutex ZeCommandListCacheMutex;

  raii::ze_command_list_handle_t
  getCommandList(const command_list_descriptor_t &desc);
  void addCommandList(const command_list_descriptor_t &desc,
                      raii::ze_command_list_handle_t cmdList);
  raii::ze_command_list_handle_t
  createCommandList(const command_list_descriptor_t &desc);
};
} // namespace v2

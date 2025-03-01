//===--------- command_list_cache.cpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_list_cache.hpp"
#include "context.hpp"

#include "../device.hpp"

template <>
ze_structure_type_t
getZeStructureType<zex_intel_queue_copy_operations_offload_hint_exp_desc_t>() {
  return ZEX_INTEL_STRUCTURE_TYPE_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_PROPERTIES;
}

bool v2::immediate_command_list_descriptor_t::operator==(
    const immediate_command_list_descriptor_t &rhs) const {
  return ZeDevice == rhs.ZeDevice && IsInOrder == rhs.IsInOrder &&
         Mode == rhs.Mode && Priority == rhs.Priority && Index == rhs.Index;
}

bool v2::regular_command_list_descriptor_t::operator==(
    const regular_command_list_descriptor_t &rhs) const {
  return ZeDevice == rhs.ZeDevice && Ordinal == rhs.Ordinal &&
         IsInOrder == rhs.IsInOrder;
}

namespace v2 {
inline size_t command_list_descriptor_hash_t::operator()(
    const command_list_descriptor_t &desc) const {
  if (auto ImmCmdDesc =
          std::get_if<immediate_command_list_descriptor_t>(&desc)) {
    return combine_hashes(0, ImmCmdDesc->ZeDevice, ImmCmdDesc->Ordinal,
                          ImmCmdDesc->IsInOrder, ImmCmdDesc->Mode,
                          ImmCmdDesc->Priority, ImmCmdDesc->Index);
  } else {
    auto RegCmdDesc = std::get<regular_command_list_descriptor_t>(desc);
    return combine_hashes(0, RegCmdDesc.ZeDevice, RegCmdDesc.IsInOrder,
                          RegCmdDesc.Ordinal);
  }
}

command_list_cache_t::command_list_cache_t(ze_context_handle_t ZeContext,
                                           bool ZeCopyOffloadExtensionSupported)
    : ZeContext{ZeContext},
      ZeCopyOffloadExtensionSupported{ZeCopyOffloadExtensionSupported} {}

raii::ze_command_list_handle_t
command_list_cache_t::createCommandList(const command_list_descriptor_t &desc) {
  ZeStruct<zex_intel_queue_copy_operations_offload_hint_exp_desc_t> offloadDesc;
  auto requestedCopyOffload =
      std::visit([](auto &&arg) { return arg.CopyOffloadEnabled; }, desc);

  if (!ZeCopyOffloadExtensionSupported && requestedCopyOffload) {
    logger::info(
        "Copy offload is requested but is not supported by the driver.");
    offloadDesc.copyOffloadEnabled = false;
  } else {
    offloadDesc.copyOffloadEnabled = requestedCopyOffload;
  }

  if (auto ImmCmdDesc =
          std::get_if<immediate_command_list_descriptor_t>(&desc)) {
    ze_command_list_handle_t ZeCommandList;
    ZeStruct<ze_command_queue_desc_t> QueueDesc;
    QueueDesc.ordinal = ImmCmdDesc->Ordinal;
    QueueDesc.mode = ImmCmdDesc->Mode;
    QueueDesc.priority = ImmCmdDesc->Priority;
    QueueDesc.flags =
        ImmCmdDesc->IsInOrder ? ZE_COMMAND_QUEUE_FLAG_IN_ORDER : 0;
    if (ImmCmdDesc->Index.has_value()) {
      QueueDesc.flags |= ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
      QueueDesc.index = ImmCmdDesc->Index.value();
    }
    QueueDesc.pNext = &offloadDesc;

    logger::debug("create command list ordinal: {}, type: immediate, device: "
                  "{}, inOrder: {}",
                  ImmCmdDesc->Ordinal, ImmCmdDesc->ZeDevice,
                  ImmCmdDesc->IsInOrder);

    ZE2UR_CALL_THROWS(
        zeCommandListCreateImmediate,
        (ZeContext, ImmCmdDesc->ZeDevice, &QueueDesc, &ZeCommandList));
    return raii::ze_command_list_handle_t(ZeCommandList);
  } else {
    auto RegCmdDesc = std::get<regular_command_list_descriptor_t>(desc);
    ZeStruct<ze_command_list_desc_t> CmdListDesc;
    CmdListDesc.flags =
        RegCmdDesc.IsInOrder ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;
    CmdListDesc.commandQueueGroupOrdinal = RegCmdDesc.Ordinal;
    CmdListDesc.pNext = &offloadDesc;

    logger::debug("create command list ordinal: {}, type: immediate, device: "
                  "{}, inOrder: {}",
                  RegCmdDesc.Ordinal, RegCmdDesc.ZeDevice,
                  RegCmdDesc.IsInOrder);

    ze_command_list_handle_t ZeCommandList;
    ZE2UR_CALL_THROWS(zeCommandListCreate, (ZeContext, RegCmdDesc.ZeDevice,
                                            &CmdListDesc, &ZeCommandList));
    return raii::ze_command_list_handle_t(ZeCommandList);
  }
}

raii::command_list_unique_handle command_list_cache_t::getImmediateCommandList(
    ze_device_handle_t ZeDevice, bool IsInOrder, uint32_t Ordinal,
    bool CopyOffloadEnable, ze_command_queue_mode_t Mode,
    ze_command_queue_priority_t Priority, std::optional<uint32_t> Index) {
  TRACK_SCOPE_LATENCY("command_list_cache_t::getImmediateCommandList");

  immediate_command_list_descriptor_t Desc;
  Desc.ZeDevice = ZeDevice;
  Desc.Ordinal = Ordinal;
  Desc.CopyOffloadEnabled = CopyOffloadEnable;
  Desc.IsInOrder = IsInOrder;
  Desc.Mode = Mode;
  Desc.Priority = Priority;
  Desc.Index = Index;

  auto [CommandList, _] = getCommandList(Desc).release();
  return raii::command_list_unique_handle(
      CommandList, [Cache = this, Desc](ze_command_list_handle_t CmdList) {
        Cache->addCommandList(Desc, raii::ze_command_list_handle_t(CmdList));
      });
}

raii::command_list_unique_handle
command_list_cache_t::getRegularCommandList(ze_device_handle_t ZeDevice,
                                            bool IsInOrder, uint32_t Ordinal,
                                            bool CopyOffloadEnable) {
  TRACK_SCOPE_LATENCY("command_list_cache_t::getRegularCommandList");

  regular_command_list_descriptor_t Desc;
  Desc.ZeDevice = ZeDevice;
  Desc.IsInOrder = IsInOrder;
  Desc.Ordinal = Ordinal;
  Desc.CopyOffloadEnabled = CopyOffloadEnable;

  auto [CommandList, _] = getCommandList(Desc).release();

  return raii::command_list_unique_handle(
      CommandList, [Cache = this, Desc](ze_command_list_handle_t CmdList) {
        Cache->addCommandList(Desc, raii::ze_command_list_handle_t(CmdList));
      });
}

raii::ze_command_list_handle_t
command_list_cache_t::getCommandList(const command_list_descriptor_t &desc) {
  std::unique_lock<ur_mutex> Lock(ZeCommandListCacheMutex);
  auto it = ZeCommandListCache.find(desc);
  if (it == ZeCommandListCache.end()) {
    Lock.unlock();
    return createCommandList(desc);
  }

  assert(!it->second.empty());

  raii::ze_command_list_handle_t CommandListHandle =
      std::move(it->second.top());
  it->second.pop();

  if (it->second.empty())
    ZeCommandListCache.erase(it);

  if (std::holds_alternative<regular_command_list_descriptor_t>(desc)) {
    ZE2UR_CALL_THROWS(zeCommandListReset, (CommandListHandle.get()));
  }

  return CommandListHandle;
}

void command_list_cache_t::addCommandList(
    const command_list_descriptor_t &desc,
    raii::ze_command_list_handle_t cmdList) {
  // TODO: add a limit?
  std::unique_lock<ur_mutex> Lock(ZeCommandListCacheMutex);
  auto [it, _] = ZeCommandListCache.try_emplace(desc);
  it->second.emplace(std::move(cmdList));
}

size_t command_list_cache_t::getNumImmediateCommandLists() {
  std::unique_lock<ur_mutex> Lock(ZeCommandListCacheMutex);
  size_t NumLists = 0;
  for (auto &Pair : ZeCommandListCache) {
    if (std::holds_alternative<immediate_command_list_descriptor_t>(Pair.first))
      NumLists += Pair.second.size();
  }
  return NumLists;
}

size_t command_list_cache_t::getNumRegularCommandLists() {
  std::unique_lock<ur_mutex> Lock(ZeCommandListCacheMutex);
  size_t NumLists = 0;
  for (auto &Pair : ZeCommandListCache) {
    if (std::holds_alternative<regular_command_list_descriptor_t>(Pair.first))
      NumLists += Pair.second.size();
  }
  return NumLists;
}

} // namespace v2

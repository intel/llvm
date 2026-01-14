//===----------- memory.hpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ur_api.h"

#include "common.hpp"

struct BufferMem {
  enum class AllocMode {
    Default,
    CopyIn,
    AllocHostPtr,
  };

  struct BufferMap {
    size_t MapSize;
    size_t MapOffset;
    ur_map_flags_t MapFlags;
    // Allocated host memory used exclusively for this map.
    std::unique_ptr<unsigned char[]> MapMem;

    BufferMap(size_t MapSize, size_t MapOffset, ur_map_flags_t MapFlags)
        : MapSize(MapSize), MapOffset(MapOffset), MapFlags(MapFlags),
          MapMem(nullptr) {}

    BufferMap(size_t MapSize, size_t MapOffset, ur_map_flags_t MapFlags,
              std::unique_ptr<unsigned char[]> &&MapMem)
        : MapSize(MapSize), MapOffset(MapOffset), MapFlags(MapFlags),
          MapMem(std::move(MapMem)) {}
  };

  ur_mem_handle_t Parent;
  // Underlying device pointer
  void *Ptr;
  // Pointer associated with this device on the host
  void *HostPtr;
  size_t Size;

  AllocMode MemAllocMode;
  std::unordered_map<void *, BufferMap> PtrToBufferMap;

  BufferMem(ur_mem_handle_t Parent, BufferMem::AllocMode Mode, void *Ptr,
            void *HostPtr, size_t Size)
      : Parent{Parent}, Ptr{Ptr}, HostPtr{HostPtr}, Size{Size},
        MemAllocMode{Mode} {};

  void *get() const noexcept { return Ptr; }
  size_t getSize() const noexcept { return Size; }

  BufferMap *getMapDetails(void *Map) {
    auto Details = PtrToBufferMap.find(Map);
    if (Details != PtrToBufferMap.end()) {
      return &Details->second;
    }
    return nullptr;
  }

  void *mapToPtr(size_t MapSize, size_t MapOffset,
                 ur_map_flags_t MapFlags) noexcept {

    void *MapPtr = nullptr;
    // If the buffer already has a host pointer we can just use it, otherwise
    // create a new host allocation
    if (HostPtr == nullptr) {
      auto MapMem = std::make_unique<unsigned char[]>(MapSize);
      MapPtr = MapMem.get();
      PtrToBufferMap.insert(
          {MapPtr, BufferMap(MapSize, MapOffset, MapFlags, std::move(MapMem))});
    } else {
      MapPtr = static_cast<char *>(HostPtr) + MapOffset;
      PtrToBufferMap.insert({MapPtr, BufferMap(MapSize, MapOffset, MapFlags)});
    }
    return MapPtr;
  }

  void unmap(void *MapPtr) noexcept {
    assert(MapPtr != nullptr);
    PtrToBufferMap.erase(MapPtr);
  }
};

struct ur_mem_handle_t_ : RefCounted {
  ur_context_handle_t Context;

  ur_mem_flags_t MemFlags;
  bool IsNativeHandleOwned;

  // For now we only support BufferMem. Eventually we'll support images, so use
  // a variant to store the underlying object.
  std::variant<BufferMem> Mem;

  ur_mem_handle_t_(ur_context_handle_t Context, ur_mem_handle_t Parent,
                   ur_mem_flags_t MemFlags, BufferMem::AllocMode Mode,
                   void *Ptr, void *HostPtr, size_t Size)
      : Context{Context}, MemFlags{MemFlags}, IsNativeHandleOwned(true),
        Mem{BufferMem{Parent, Mode, Ptr, HostPtr, Size}} {
    urContextRetain(Context);
  };

  ~ur_mem_handle_t_() { urContextRelease(Context); }

  ur_context_handle_t getContext() const noexcept { return Context; }

  BufferMem *AsBufferMem() noexcept {
    if (std::holds_alternative<BufferMem>(Mem)) {
      return &std::get<BufferMem>(Mem);
    }
    return nullptr;
  }
};

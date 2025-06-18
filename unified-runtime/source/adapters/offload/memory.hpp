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
    UseHostPtr,
    CopyIn,
    AllocHostPtr,
  };

  ur_mem_handle_t Parent;
  // Underlying device pointer
  void *Ptr;
  // Pointer associated with this device on the host
  void *HostPtr;
  size_t Size;

  AllocMode MemAllocMode;

  BufferMem(ur_mem_handle_t Parent, BufferMem::AllocMode Mode, void *Ptr,
            void *HostPtr, size_t Size)
      : Parent{Parent}, Ptr{Ptr}, HostPtr{HostPtr}, Size{Size},
        MemAllocMode{Mode} {};

  void *get() const noexcept { return Ptr; }
  size_t getSize() const noexcept { return Size; }
};

struct ur_mem_handle_t_ : RefCounted {
  ur_context_handle_t Context;

  enum class Type { Buffer } MemType;
  ur_mem_flags_t MemFlags;

  // For now we only support BufferMem. Eventually we'll support images, so use
  // a variant to store the underlying object.
  std::variant<BufferMem> Mem;

  ur_mem_handle_t_(ur_context_handle_t Context, ur_mem_handle_t Parent,
                   ur_mem_flags_t MemFlags, BufferMem::AllocMode Mode,
                   void *Ptr, void *HostPtr, size_t Size)
      : Context{Context}, MemType{Type::Buffer}, MemFlags{MemFlags},
        Mem{BufferMem{Parent, Mode, Ptr, HostPtr, Size}} {
    urContextRetain(Context);
  };

  ~ur_mem_handle_t_() { urContextRelease(Context); }

  ur_context_handle_t getContext() const noexcept { return Context; }
};

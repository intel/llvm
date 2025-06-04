//===-------------- memory.hpp - Native CPU Adapter -----------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
#include "context.hpp"

struct ur_mem_handle_t_ : ur_object {
  ur_mem_handle_t_(size_t Size, bool _IsImage)
      : _mem{static_cast<char *>(malloc(Size))}, _ownsMem{true},
        IsImage{_IsImage} {}

  ur_mem_handle_t_(void *HostPtr, size_t Size, bool _IsImage)
      : _mem{static_cast<char *>(malloc(Size))}, _ownsMem{true},
        IsImage{_IsImage} {
    memcpy(_mem, HostPtr, Size);
  }

  ur_mem_handle_t_(void *HostPtr, bool _IsImage)
      : _mem{static_cast<char *>(HostPtr)}, _ownsMem{false}, IsImage{_IsImage} {
  }

  ~ur_mem_handle_t_() {
    if (_ownsMem) {
      free(_mem);
    }
  }

  // Method to get type of the derived object (image or buffer)
  bool isImage() const { return this->IsImage; }

  char *_mem;
  bool _ownsMem;

private:
  const bool IsImage;
};

struct ur_buffer final : ur_mem_handle_t_ {
  // Buffer constructor
  ur_buffer(ur_context_handle_t /* Context*/, void *HostPtr)
      : ur_mem_handle_t_(HostPtr, false) {}
  ur_buffer(ur_context_handle_t /* Context*/, void *HostPtr, size_t Size)
      : ur_mem_handle_t_(HostPtr, Size, false) {}
  ur_buffer(ur_context_handle_t /* Context*/, size_t Size)
      : ur_mem_handle_t_(Size, false) {}
  ur_buffer(ur_buffer *b, size_t Offset, size_t /*Size*/)
      : ur_mem_handle_t_(b->_mem + Offset, false), SubBuffer(b) {
    SubBuffer.Origin = Offset;
  }

  bool isSubBuffer() const { return SubBuffer.Parent != nullptr; }

  struct BB {
    BB(ur_buffer *b) : Parent(b), Origin(0) {}
    BB() : BB(nullptr) {}
    ur_buffer *const Parent;
    size_t Origin; // only valid if Parent != nullptr
  } SubBuffer;
};

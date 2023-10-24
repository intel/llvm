//===-------------- memory.hpp - Native CPU Adapter -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
#include "context.hpp"

struct ur_mem_handle_t_ : _ur_object {
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

  void decrementRefCount() noexcept { _refCount--; }

  // Method to get type of the derived object (image or buffer)
  bool isImage() const { return this->IsImage; }

  char *_mem;
  bool _ownsMem;
  std::atomic_uint32_t _refCount = {1};

private:
  const bool IsImage;
};

struct _ur_buffer final : ur_mem_handle_t_ {
  // Buffer constructor
  _ur_buffer(ur_context_handle_t /* Context*/, void *HostPtr)
      : ur_mem_handle_t_(HostPtr, false) {}
  _ur_buffer(ur_context_handle_t /* Context*/, void *HostPtr, size_t Size)
      : ur_mem_handle_t_(HostPtr, Size, false) {}
  _ur_buffer(ur_context_handle_t /* Context*/, size_t Size)
      : ur_mem_handle_t_(Size, false) {}
  _ur_buffer(_ur_buffer *b, size_t Offset, size_t Size)
      : ur_mem_handle_t_(b->_mem + Offset, false), SubBuffer(b) {
    SubBuffer.Origin = Offset;
  }

  bool isSubBuffer() const { return SubBuffer.Parent != nullptr; }

  struct BB {
    BB(_ur_buffer *b) : Parent(b) {}
    BB() : BB(nullptr) {}
    _ur_buffer *const Parent;
    size_t Origin; // only valid if Parent != nullptr
  } SubBuffer;
};

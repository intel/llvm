//===--------- program.hpp - Native CPU Adapter ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include "context.hpp"
#include <map>

struct ur_program_handle_t_ {
  ur_program_handle_t_(ur_context_handle_t ctx, const unsigned char *pBinary)
      : _ctx{ctx}, _ptr{pBinary}, _refCount{1} {}

  uint32_t getReferenceCount() const noexcept { return _refCount; }

  ur_context_handle_t _ctx;
  const unsigned char *_ptr;
  std::atomic_uint32_t _refCount;

  struct _compare {
    bool operator()(char const *a, char const *b) const {
      return std::strcmp(a, b) < 0;
    }
  };

  std::map<const char *, const unsigned char *, _compare> _kernels;
};

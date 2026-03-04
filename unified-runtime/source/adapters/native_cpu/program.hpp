//===--------- program.hpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include "context.hpp"

#include <array>
#include <map>

namespace native_cpu {
using WGSize_t = std::array<uint32_t, 3>;
}

struct ur_program_handle_t_ : RefCounted {
  ur_program_handle_t_(ur_context_handle_t ctx, const unsigned char *pBinary)
      : _ctx{ctx}, _ptr{pBinary} {}

  uint32_t getReferenceCount() const noexcept { return _refCount; }

  ur_context_handle_t _ctx;
  const unsigned char *_ptr;
  struct _compare {
    bool operator()(char const *a, char const *b) const {
      return std::strcmp(a, b) < 0;
    }
  };

  std::map<const char *, const unsigned char *, _compare> _kernels;
  std::unordered_map<std::string, native_cpu::WGSize_t>
      KernelReqdWorkGroupSizeMD;
  std::unordered_map<std::string, native_cpu::WGSize_t>
      KernelMaxWorkGroupSizeMD;
  std::unordered_map<std::string, uint64_t> KernelMaxLinearWorkGroupSizeMD;
};

// The nativecpu_entry struct is also defined as LLVM-IR in the
// clang-offload-wrapper tool. The two definitions need to match,
// therefore any change to this struct needs to be reflected in the
// offload-wrapper.
struct nativecpu_entry {
  const char *kernelname;
  const unsigned char *kernel_ptr;
};

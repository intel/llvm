//==---------------- free_function_kernel_args_impl.hpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Source-side definition of FreeFunctionArgsStorage, the object that owns the
// explicit arguments collected for a free function kernel submitted directly to
// a queue (see sycl/detail/free_function_kernel_args.hpp for the header-visible
// collector). It is defined here rather than in a single translation unit so
// that both the collector implementation and the queue submission path can
// operate on a complete type.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/kernel_arg_desc.hpp>

#include <sycl/detail/kernel_desc.hpp>

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Owns the collected kernel arguments together with the storage that keeps the
// argument data alive. The storage members mirror the corresponding members of
// CG::StorageInitHelper so the runtime can move them into a command group.
class FreeFunctionArgsStorage {
public:
  // The collected kernel arguments. Pointers in these descriptors reference
  // MArgsStorage / MSharedPtrStorage below and stay valid for as long as this
  // object (or the storage moved out of it) lives.
  std::vector<detail::ArgDesc> MArgs;

  // Storage for standard-layout / pointer / raw argument bytes. A vector of
  // vectors is used so that adding a new argument never invalidates the data
  // pointer of a previously stored one (the heap buffer of each inner vector is
  // preserved when the outer vector reallocates).
  std::vector<std::vector<char>> MArgsStorage;

  // Storage for arguments whose descriptors must point at a live SYCL object
  // (work_group_memory_impl, sampler).
  std::vector<std::shared_ptr<const void>> MSharedPtrStorage;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

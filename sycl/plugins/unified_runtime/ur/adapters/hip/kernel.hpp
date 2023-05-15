//===--------- kernel.hpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

#include <atomic>
#include <cassert>
#include <numeric>

#include "program.hpp"

struct ur_kernel_handle_t_ : _ur_object {
  using native_type = hipFunction_t;

  native_type function_;
  native_type functionWithOffsetParam_;
  std::string name_;
  ur_context_handle_t context_;
  ur_program_handle_t program_;
  std::atomic_uint32_t refCount_;

  /// Structure that holds the arguments to the kernel.
  /// Note earch argument size is known, since it comes
  /// from the kernel signature.
  /// This is not something can be queried from the HIP API
  /// so there is a hard-coded size (\ref MAX_PARAM_BYTES)
  /// and a storage.
  ///
  struct arguments {
    static constexpr size_t MAX_PARAM_BYTES = 4000u;
    using args_t = std::array<char, MAX_PARAM_BYTES>;
    using args_size_t = std::vector<size_t>;
    using args_index_t = std::vector<void *>;
    args_t storage_;
    args_size_t paramSizes_;
    args_index_t indices_;
    args_size_t offsetPerIndex_;

    std::uint32_t implicitOffsetArgs_[3] = {0, 0, 0};

    arguments() {
      // Place the implicit offset index at the end of the indicies collection
      indices_.emplace_back(&implicitOffsetArgs_);
    }

    /// Adds an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    /// Implicit offset argument is kept at the back of the indices collection.
    void add_arg(size_t index, size_t size, const void *arg,
                 size_t localSize = 0) {
      if (index + 2 > indices_.size()) {
        // Move implicit offset argument index with the end
        indices_.resize(index + 2, indices_.back());
        // Ensure enough space for the new argument
        paramSizes_.resize(index + 1);
        offsetPerIndex_.resize(index + 1);
      }
      paramSizes_[index] = size;
      // calculate the insertion point on the array
      size_t insertPos = std::accumulate(std::begin(paramSizes_),
                                         std::begin(paramSizes_) + index, 0);
      // Update the stored value for the argument
      std::memcpy(&storage_[insertPos], arg, size);
      indices_[index] = &storage_[insertPos];
      offsetPerIndex_[index] = localSize;
    }

    void add_local_arg(size_t index, size_t size) {
      size_t localOffset = this->get_local_size();

      // maximum required alignment is the size of the largest vector type
      const size_t max_alignment = sizeof(double) * 16;

      // for arguments smaller than the maximum alignment simply align to the
      // size of the argument
      const size_t alignment = std::min(max_alignment, size);

      // align the argument
      size_t alignedLocalOffset = localOffset;
      if (localOffset % alignment != 0) {
        alignedLocalOffset += alignment - (localOffset % alignment);
      }

      add_arg(index, sizeof(size_t), (const void *)&(alignedLocalOffset),
              size + (alignedLocalOffset - localOffset));
    }

    void set_implicit_offset(size_t size, std::uint32_t *implicitOffset) {
      assert(size == sizeof(std::uint32_t) * 3);
      std::memcpy(implicitOffsetArgs_, implicitOffset, size);
    }

    void clear_local_size() {
      std::fill(std::begin(offsetPerIndex_), std::end(offsetPerIndex_), 0);
    }

    const args_index_t &get_indices() const noexcept { return indices_; }

    uint32_t get_local_size() const {
      return std::accumulate(std::begin(offsetPerIndex_),
                             std::end(offsetPerIndex_), 0);
    }
  } args_;

  ur_kernel_handle_t_(hipFunction_t func, hipFunction_t funcWithOffsetParam,
                      const char *name, ur_program_handle_t program,
                      ur_context_handle_t ctxt)
      : function_{func}, functionWithOffsetParam_{funcWithOffsetParam},
        name_{name}, context_{ctxt}, program_{program}, refCount_{1} {
    urProgramRetain(program_);
    urContextRetain(context_);
  }

  ur_kernel_handle_t_(hipFunction_t func, const char *name,
                      ur_program_handle_t program, ur_context_handle_t ctxt)
      : ur_kernel_handle_t_{func, nullptr, name, program, ctxt} {}

  ~ur_kernel_handle_t_() {
    urProgramRelease(program_);
    urContextRelease(context_);
  }

  ur_program_handle_t get_program() const noexcept { return program_; }

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

  native_type get() const noexcept { return function_; };

  native_type get_with_offset_parameter() const noexcept {
    return functionWithOffsetParam_;
  };

  bool has_with_offset_parameter() const noexcept {
    return functionWithOffsetParam_ != nullptr;
  }

  ur_context_handle_t get_context() const noexcept { return context_; };

  const char *get_name() const noexcept { return name_.c_str(); }

  /// Returns the number of arguments, excluding the implicit global offset.
  /// Note this only returns the current known number of arguments, not the
  /// real one required by the kernel, since this cannot be queried from
  /// the HIP Driver API
  uint32_t get_num_args() const noexcept { return args_.indices_.size() - 1; }

  void set_kernel_arg(int index, size_t size, const void *arg) {
    args_.add_arg(index, size, arg);
  }

  void set_kernel_local_arg(int index, size_t size) {
    args_.add_local_arg(index, size);
  }

  void set_implicit_offset_arg(size_t size, std::uint32_t *implicitOffset) {
    return args_.set_implicit_offset(size, implicitOffset);
  }

  const arguments::args_index_t &get_arg_indices() const {
    return args_.get_indices();
  }

  uint32_t get_local_size() const noexcept { return args_.get_local_size(); }

  void clear_local_size() { args_.clear_local_size(); }
};

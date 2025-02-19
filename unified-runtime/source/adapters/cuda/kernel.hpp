//===--------- kernel.hpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <ur_api.h>

#include <array>
#include <atomic>
#include <cassert>
#include <numeric>

#include "program.hpp"

/// Implementation of a UR Kernel for CUDA
///
/// UR Kernels are used to set kernel arguments,
/// creating a state on the Kernel object for a given
/// invocation. This is not the case of CUFunction objects,
/// which are simply passed together with the arguments on the invocation.
/// The UR Kernel implementation for CUDA stores the list of arguments,
/// argument sizes, and offsets to emulate the interface of UR Kernel,
/// saving the arguments for the later dispatch.
/// Note that in UR API, the Local memory is specified as a size per
/// individual argument, but in CUDA only the total usage of shared
/// memory is required since it is not passed as a parameter.
/// A compiler pass converts the UR API local memory model into the
/// CUDA shared model. This object simply calculates the total of
/// shared memory, and the initial offsets of each parameter.
struct ur_kernel_handle_t_ {
  using native_type = CUfunction;

  native_type Function;
  native_type FunctionWithOffsetParam;
  std::string Name;
  ur_context_handle_t Context;
  ur_program_handle_t Program;
  std::atomic_uint32_t RefCount;

  static constexpr uint32_t ReqdThreadsPerBlockDimensions = 3u;
  size_t ReqdThreadsPerBlock[ReqdThreadsPerBlockDimensions];
  size_t MaxThreadsPerBlock[ReqdThreadsPerBlockDimensions];
  size_t MaxLinearThreadsPerBlock{0};
  int RegsPerThread{0};

  /// Structure that holds the arguments to the kernel.
  /// Note each argument size is known, since it comes
  /// from the kernel signature.
  /// This is not something can be queried from the CUDA API
  /// so there is a hard-coded size (\ref MAX_PARAM_BYTES)
  /// and a storage.
  struct arguments {
    static constexpr size_t MaxParamBytes = 4000u;
    using args_t = std::array<char, MaxParamBytes>;
    using args_size_t = std::vector<size_t>;
    using args_index_t = std::vector<void *>;
    /// Storage shared by all args which is mem copied into when adding a new
    /// argument.
    args_t Storage;
    /// Aligned size of each parameter, including padding.
    args_size_t ParamSizes;
    /// Byte offset into /p Storage allocation for each argument.
    args_index_t ArgPointers;
    /// Position in the Storage array where the next argument should added.
    size_t InsertPos = 0;
    /// Aligned size in bytes for each local memory parameter after padding has
    /// been added. Zero if the argument at the index isn't a local memory
    /// argument.
    args_size_t AlignedLocalMemSize;
    /// Original size in bytes for each local memory parameter, prior to being
    /// padded to appropriate alignment. Zero if the argument at the index
    /// isn't a local memory argument.
    args_size_t OriginalLocalMemSize;
    size_t WorkGroupMemory = 0;

    // A struct to keep track of memargs so that we can do dependency analysis
    // at urEnqueueKernelLaunch
    struct mem_obj_arg {
      ur_mem_handle_t_ *Mem;
      int Index;
      ur_mem_flags_t AccessFlags;
    };
    std::vector<mem_obj_arg> MemObjArgs;

    std::uint32_t ImplicitOffsetArgs[3] = {0, 0, 0};

    arguments() {
      // Place the implicit offset index at the end of the ArgPointers
      // collection.
      ArgPointers.emplace_back(&ImplicitOffsetArgs);
    }

    /// Add an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    /// Implicit offset argument is kept at the back of the ArgPointers
    /// collection.
    void addArg(size_t Index, size_t Size, const void *Arg,
                size_t LocalSize = 0) {
      // Expand storage to accommodate this Index if needed.
      if (Index + 2 > ArgPointers.size()) {
        // Move implicit offset argument index with the end
        ArgPointers.resize(Index + 2, ArgPointers.back());
        // Ensure enough space for the new argument
        ParamSizes.resize(Index + 1);
        AlignedLocalMemSize.resize(Index + 1);
        OriginalLocalMemSize.resize(Index + 1);
      }

      // Copy new argument to storage if it hasn't been added before.
      if (ParamSizes[Index] == 0) {
        ParamSizes[Index] = Size;
        std::memcpy(&Storage[InsertPos], Arg, Size);
        ArgPointers[Index] = &Storage[InsertPos];
        AlignedLocalMemSize[Index] = LocalSize;
        InsertPos += Size;
      }
      // Otherwise, update the existing argument.
      else {
        std::memcpy(ArgPointers[Index], Arg, Size);
        AlignedLocalMemSize[Index] = LocalSize;
        assert(Size == ParamSizes[Index]);
      }
    }

    /// Returns the padded size and offset of a local memory argument.
    /// Local memory arguments need to be padded if the alignment for the size
    /// doesn't match the current offset into the kernel local data.
    /// @param Index Kernel arg index.
    /// @param Size User passed size of local parameter.
    /// @return Tuple of (Aligned size, Aligned offset into local data).
    std::pair<size_t, size_t> calcAlignedLocalArgument(size_t Index,
                                                       size_t Size) {
      // Store the unpadded size of the local argument
      if (Index + 2 > ArgPointers.size()) {
        AlignedLocalMemSize.resize(Index + 1);
        OriginalLocalMemSize.resize(Index + 1);
      }
      OriginalLocalMemSize[Index] = Size;

      // Calculate the current starting offset into local data
      const size_t LocalOffset =
          std::accumulate(std::begin(AlignedLocalMemSize),
                          std::next(std::begin(AlignedLocalMemSize), Index),
                          size_t{WorkGroupMemory});

      // Maximum required alignment is the size of the largest vector type
      const size_t MaxAlignment = sizeof(double) * 16;

      // For arguments smaller than the maximum alignment simply align to the
      // size of the argument
      const size_t Alignment = std::min(MaxAlignment, Size);

      // Align the argument
      size_t AlignedLocalOffset = LocalOffset;
      const size_t Pad = LocalOffset % Alignment;
      if (Pad != 0) {
        AlignedLocalOffset += Alignment - Pad;
      }

      const size_t AlignedLocalSize = Size + (AlignedLocalOffset - LocalOffset);
      return std::make_pair(AlignedLocalSize, AlignedLocalOffset);
    }

    // Iterate over each existing local argument which follows StartIndex
    // index, update the offset and pointer into the kernel local memory.
    void updateLocalArgOffset(size_t StartIndex) {
      const size_t NumArgs =
          ArgPointers.size() - 1; // Accounts for implicit arg
      for (auto SuccIndex = StartIndex; SuccIndex < NumArgs; SuccIndex++) {
        const size_t OriginalLocalSize = OriginalLocalMemSize[SuccIndex];
        if (OriginalLocalSize == 0) {
          // Skip if successor argument isn't a local memory arg
          continue;
        }

        // Recalculate alignment
        auto [SuccAlignedLocalSize, SuccAlignedLocalOffset] =
            calcAlignedLocalArgument(SuccIndex, OriginalLocalSize);

        // Store new local memory size
        AlignedLocalMemSize[SuccIndex] = SuccAlignedLocalSize;

        // Store new offset into local data
        std::memcpy(ArgPointers[SuccIndex], &SuccAlignedLocalOffset,
                    sizeof(size_t));
      }
    }

    void addLocalArg(size_t Index, size_t Size) {
      // Get the aligned argument size and offset into local data
      auto [AlignedLocalSize, AlignedLocalOffset] =
          calcAlignedLocalArgument(Index, Size);

      // Store argument details
      addArg(Index, sizeof(size_t), (const void *)&(AlignedLocalOffset),
             AlignedLocalSize);
      // For every existing local argument which follows at later argument
      // indices, update the offset and pointer into the kernel local memory.
      // Required as padding will need to be recalculated.
      updateLocalArgOffset(Index + 1);
    }

    void addMemObjArg(int Index, ur_mem_handle_t hMem, ur_mem_flags_t Flags) {
      assert(hMem && "Invalid mem handle");
      // To avoid redundancy we are not storing mem obj with index i at index
      // i in the vec of MemObjArgs.
      for (auto &Arg : MemObjArgs) {
        if (Arg.Index == Index) {
          // Overwrite the mem obj with the same index
          Arg = arguments::mem_obj_arg{hMem, Index, Flags};
          return;
        }
      }
      MemObjArgs.push_back(arguments::mem_obj_arg{hMem, Index, Flags});
    }

    void setWorkGroupMemory(size_t MemSize) {
      // If the WorkGroupMemory is the same as MemSize, then all accessors
      // offsets accounted for this extra memory
      if (WorkGroupMemory == MemSize)
        return;
      WorkGroupMemory = MemSize;
      // Update local accessor offsets
      updateLocalArgOffset(/*StartIndex=*/0);
    }

    void setImplicitOffset(size_t Size, std::uint32_t *ImplicitOffset) {
      assert(Size == sizeof(std::uint32_t) * 3);
      std::memcpy(ImplicitOffsetArgs, ImplicitOffset, Size);
    }

    const args_index_t &getArgPointers() const noexcept { return ArgPointers; }

    uint32_t getLocalSize() const {
      return std::accumulate(std::begin(AlignedLocalMemSize),
                             std::end(AlignedLocalMemSize), 0) +
             WorkGroupMemory;
    }
  } Args;

  ur_kernel_handle_t_(CUfunction Func, CUfunction FuncWithOffsetParam,
                      const char *Name, ur_program_handle_t Program,
                      ur_context_handle_t Context)
      : Function{Func}, FunctionWithOffsetParam{FuncWithOffsetParam},
        Name{Name}, Context{Context}, Program{Program}, RefCount{1} {
    urProgramRetain(Program);
    urContextRetain(Context);
    /// Note: this code assumes that there is only one device per context
    ur_result_t RetError = urKernelGetGroupInfo(
        this, Program->getDevice(),
        UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(ReqdThreadsPerBlock), ReqdThreadsPerBlock, nullptr);
    (void)RetError;
    assert(RetError == UR_RESULT_SUCCESS);
    /// Note: this code assumes that there is only one device per context
    RetError = urKernelGetGroupInfo(
        this, Program->getDevice(),
        UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE,
        sizeof(MaxThreadsPerBlock), MaxThreadsPerBlock, nullptr);
    assert(RetError == UR_RESULT_SUCCESS);
    /// Note: this code assumes that there is only one device per context
    RetError = urKernelGetGroupInfo(
        this, Program->getDevice(),
        UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE,
        sizeof(MaxLinearThreadsPerBlock), &MaxLinearThreadsPerBlock, nullptr);
    assert(RetError == UR_RESULT_SUCCESS);
    UR_CHECK_ERROR(
        cuFuncGetAttribute(&RegsPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS, Func));
  }

  ~ur_kernel_handle_t_() {
    urProgramRelease(Program);
    urContextRelease(Context);
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  native_type get() const noexcept { return Function; };

  ur_program_handle_t getProgram() const noexcept { return Program; };

  native_type get_with_offset_parameter() const noexcept {
    return FunctionWithOffsetParam;
  };

  bool has_with_offset_parameter() const noexcept {
    return FunctionWithOffsetParam != nullptr;
  }

  ur_context_handle_t getContext() const noexcept { return Context; };

  const char *getName() const noexcept { return Name.c_str(); }

  /// Get the number of kernel arguments, excluding the implicit global offset.
  /// Note this only returns the current known number of arguments, not the
  /// real one required by the kernel, since this cannot be queried from
  /// the CUDA Driver API
  uint32_t getNumArgs() const noexcept {
    return static_cast<uint32_t>(Args.ArgPointers.size() - 1);
  }

  void setKernelArg(int Index, size_t Size, const void *Arg) {
    Args.addArg(Index, Size, Arg);
  }

  void setKernelLocalArg(int Index, size_t Size) {
    Args.addLocalArg(Index, Size);
  }

  void setImplicitOffsetArg(size_t Size, std::uint32_t *ImplicitOffset) {
    return Args.setImplicitOffset(Size, ImplicitOffset);
  }

  const arguments::args_index_t &getArgPointers() const {
    return Args.getArgPointers();
  }

  void setWorkGroupMemory(size_t MemSize) { Args.setWorkGroupMemory(MemSize); }
  uint32_t getLocalSize() const noexcept { return Args.getLocalSize(); }

  size_t getRegsPerThread() const noexcept { return RegsPerThread; };
};

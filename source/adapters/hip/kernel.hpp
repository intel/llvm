//===--------- kernel.hpp - HIP Adapter -----------------------------------===//
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

#include <atomic>
#include <cassert>
#include <numeric>

#include "program.hpp"

/// Implementation of a UR Kernel for HIP
///
/// UR Kernels are used to set kernel arguments,
/// creating a state on the Kernel object for a given
/// invocation. This is not the case of HIPFunction objects,
/// which are simply passed together with the arguments on the invocation.
/// The UR Kernel implementation for HIP stores the list of arguments,
/// argument sizes, and offsets to emulate the interface of UR Kernel,
/// saving the arguments for the later dispatch.
/// Note that in UR API, the Local memory is specified as a size per
/// individual argument, but in HIP only the total usage of shared
/// memory is required since it is not passed as a parameter.
/// A compiler pass converts the UR API local memory model into the
/// HIP shared model. This object simply calculates the total of
/// shared memory, and the initial offsets of each parameter.
struct ur_kernel_handle_t_ {
  using native_type = hipFunction_t;

  native_type Function;
  native_type FunctionWithOffsetParam;
  std::string Name;
  ur_context_handle_t Context;
  ur_program_handle_t Program;
  std::atomic_uint32_t RefCount;

  static constexpr uint32_t ReqdThreadsPerBlockDimensions = 3u;
  size_t ReqdThreadsPerBlock[ReqdThreadsPerBlockDimensions];

  /// Structure that holds the arguments to the kernel.
  /// Note earch argument size is known, since it comes
  /// from the kernel signature.
  /// This is not something can be queried from the HIP API
  /// so there is a hard-coded size (\ref MAX_PARAM_BYTES)
  /// and a storage.
  struct arguments {
    static constexpr size_t MAX_PARAM_BYTES = 4000u;
    using args_t = std::array<char, MAX_PARAM_BYTES>;
    using args_size_t = std::vector<size_t>;
    using args_index_t = std::vector<void *>;
    /// Storage shared by all args which is mem copied into when adding a new
    /// argument.
    args_t Storage;
    /// Aligned size of each parameter, including padding.
    args_size_t ParamSizes;
    /// Byte offset into /p Storage allocation for each parameter.
    args_index_t Indices;
    /// Aligned size in bytes for each local memory parameter after padding has
    /// been added. Zero if the argument at the index isn't a local memory
    /// argument.
    args_size_t AlignedLocalMemSize;
    /// Original size in bytes for each local memory parameter, prior to being
    /// padded to appropriate alignment. Zero if the argument at the index
    /// isn't a local memory argument.
    args_size_t OriginalLocalMemSize;

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
      // Place the implicit offset index at the end of the indicies collection
      Indices.emplace_back(&ImplicitOffsetArgs);
    }

    /// Add an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    /// Implicit offset argument is kept at the back of the indices collection.
    void addArg(size_t Index, size_t Size, const void *Arg,
                size_t LocalSize = 0) {
      if (Index + 2 > Indices.size()) {
        // Move implicit offset argument Index with the end
        Indices.resize(Index + 2, Indices.back());
        // Ensure enough space for the new argument
        ParamSizes.resize(Index + 1);
        AlignedLocalMemSize.resize(Index + 1);
        OriginalLocalMemSize.resize(Index + 1);
      }
      ParamSizes[Index] = Size;
      // calculate the insertion point on the array
      size_t InsertPos = std::accumulate(std::begin(ParamSizes),
                                         std::begin(ParamSizes) + Index, 0);
      // Update the stored value for the argument
      std::memcpy(&Storage[InsertPos], Arg, Size);
      Indices[Index] = &Storage[InsertPos];
      AlignedLocalMemSize[Index] = LocalSize;
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
      if (Index + 2 > Indices.size()) {
        AlignedLocalMemSize.resize(Index + 1);
        OriginalLocalMemSize.resize(Index + 1);
      }
      OriginalLocalMemSize[Index] = Size;

      // Calculate the current starting offset into local data
      const size_t LocalOffset = std::accumulate(
          std::begin(AlignedLocalMemSize),
          std::next(std::begin(AlignedLocalMemSize), Index), size_t{0});

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
      const size_t NumArgs = Indices.size() - 1; // Accounts for implicit arg
      for (auto SuccIndex = Index + 1; SuccIndex < NumArgs; SuccIndex++) {
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
        const size_t InsertPos =
            std::accumulate(std::begin(ParamSizes),
                            std::begin(ParamSizes) + SuccIndex, size_t{0});
        std::memcpy(&Storage[InsertPos], &SuccAlignedLocalOffset,
                    sizeof(size_t));
      }
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

    void setImplicitOffset(size_t Size, std::uint32_t *ImplicitOffset) {
      assert(Size == sizeof(std::uint32_t) * 3);
      std::memcpy(ImplicitOffsetArgs, ImplicitOffset, Size);
    }

    const args_index_t &getIndices() const noexcept { return Indices; }

    uint32_t getLocalSize() const {
      return std::accumulate(std::begin(AlignedLocalMemSize),
                             std::end(AlignedLocalMemSize), 0);
    }
  } Args;

  ur_kernel_handle_t_(hipFunction_t Func, hipFunction_t FuncWithOffsetParam,
                      const char *Name, ur_program_handle_t Program,
                      ur_context_handle_t Ctxt)
      : Function{Func}, FunctionWithOffsetParam{FuncWithOffsetParam},
        Name{Name}, Context{Ctxt}, Program{Program}, RefCount{1} {
    assert(Program->getDevice());
    UR_CHECK_ERROR(urKernelGetGroupInfo(
        this, Program->getDevice(),
        UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(ReqdThreadsPerBlock), ReqdThreadsPerBlock, nullptr));
    urProgramRetain(Program);
    urContextRetain(Context);
  }

  ur_kernel_handle_t_(hipFunction_t Func, const char *Name,
                      ur_program_handle_t Program, ur_context_handle_t Ctxt)
      : ur_kernel_handle_t_{Func, nullptr, Name, Program, Ctxt} {}

  ~ur_kernel_handle_t_() {
    urProgramRelease(Program);
    urContextRelease(Context);
  }

  ur_program_handle_t getProgram() const noexcept { return Program; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  native_type get() const noexcept { return Function; };

  native_type getWithOffsetParameter() const noexcept {
    return FunctionWithOffsetParam;
  };

  bool hasWithOffsetParameter() const noexcept {
    return FunctionWithOffsetParam != nullptr;
  }

  ur_context_handle_t getContext() const noexcept { return Context; };

  const char *getName() const noexcept { return Name.c_str(); }

  /// Get the number of kernel arguments, excluding the implicit global
  /// offset. Note this only returns the current known number of arguments,
  /// not the real one required by the kernel, since this cannot be queried
  /// from the HIP Driver API
  uint32_t getNumArgs() const noexcept { return Args.Indices.size() - 1; }

  void setKernelArg(int Index, size_t Size, const void *Arg) {
    Args.addArg(Index, Size, Arg);
  }

  void setKernelLocalArg(int Index, size_t Size) {
    Args.addLocalArg(Index, Size);
  }

  void setImplicitOffsetArg(size_t Size, std::uint32_t *ImplicitOffset) {
    return Args.setImplicitOffset(Size, ImplicitOffset);
  }

  const arguments::args_index_t &getArgIndices() const {
    return Args.getIndices();
  }

  uint32_t getLocalSize() const noexcept { return Args.getLocalSize(); }
};

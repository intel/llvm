//===--------- kernel.hpp - HIP Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

#include <atomic>
#include <cassert>
#include <numeric>
#include <set>

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
    args_t Storage;
    args_size_t ParamSizes;
    args_index_t Indices;
    args_size_t OffsetPerIndex;
    std::set<const void *> PtrArgs;

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
        OffsetPerIndex.resize(Index + 1);
      }
      ParamSizes[Index] = Size;
      // calculate the insertion point on the array
      size_t InsertPos = std::accumulate(std::begin(ParamSizes),
                                         std::begin(ParamSizes) + Index, 0);
      // Update the stored value for the argument
      std::memcpy(&Storage[InsertPos], Arg, Size);
      Indices[Index] = &Storage[InsertPos];
      OffsetPerIndex[Index] = LocalSize;
    }

    void addLocalArg(size_t Index, size_t Size) {
      size_t LocalOffset = this->getLocalSize();

      // maximum required alignment is the size of the largest vector type
      const size_t MaxAlignment = sizeof(double) * 16;

      // for arguments smaller than the maximum alignment simply align to the
      // size of the argument
      const size_t Alignment = std::min(MaxAlignment, Size);

      // align the argument
      size_t AlignedLocalOffset = LocalOffset;
      size_t Pad = LocalOffset % Alignment;
      if (Pad != 0) {
        AlignedLocalOffset += Alignment - Pad;
      }

      addArg(Index, sizeof(size_t), (const void *)&AlignedLocalOffset,
             Size + AlignedLocalOffset - LocalOffset);
    }

    void setImplicitOffset(size_t Size, std::uint32_t *ImplicitOffset) {
      assert(Size == sizeof(std::uint32_t) * 3);
      std::memcpy(ImplicitOffsetArgs, ImplicitOffset, Size);
    }

    void clearLocalSize() {
      std::fill(std::begin(OffsetPerIndex), std::end(OffsetPerIndex), 0);
    }

    const args_index_t &getIndices() const noexcept { return Indices; }

    uint32_t getLocalSize() const {
      return std::accumulate(std::begin(OffsetPerIndex),
                             std::end(OffsetPerIndex), 0);
    }
  } Args;

  ur_kernel_handle_t_(hipFunction_t Func, hipFunction_t FuncWithOffsetParam,
                      const char *Name, ur_program_handle_t Program,
                      ur_context_handle_t Ctxt)
      : Function{Func}, FunctionWithOffsetParam{FuncWithOffsetParam},
        Name{Name}, Context{Ctxt}, Program{Program}, RefCount{1} {
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

  /// Get the number of kernel arguments, excluding the implicit global offset.
  /// Note this only returns the current known number of arguments, not the
  /// real one required by the kernel, since this cannot be queried from
  /// the HIP Driver API
  uint32_t getNumArgs() const noexcept { return Args.Indices.size() - 1; }

  void setKernelArg(int Index, size_t Size, const void *Arg) {
    Args.addArg(Index, Size, Arg);
  }

  /// We track all pointer arguments to be able to issue prefetches at enqueue
  /// time
  void setKernelPtrArg(int Index, size_t Size, const void *PtrArg) {
    Args.PtrArgs.insert(*static_cast<void *const *>(PtrArg));
    setKernelArg(Index, Size, PtrArg);
  }

  bool isPtrArg(const void *ptr) {
    return Args.PtrArgs.find(ptr) != Args.PtrArgs.end();
  }

  std::set<const void *> &getPtrArgs() { return Args.PtrArgs; }

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

  void clearLocalSize() { Args.clearLocalSize(); }

  void addMemObjArg(int Index, ur_mem_handle_t hMem, ur_mem_flags_t Flags) {
    assert(hMem && "Invalid mem handle");
    for (auto i = 0u; i < Args.MemObjArgs.size(); ++i) {
      if (Args.MemObjArgs[i].Index == Index) {
        // Overwrite the mem obj with the same index
        Args.MemObjArgs[i] = arguments::mem_obj_arg{hMem, Index, Flags};
        return;
      }
    }
    Args.MemObjArgs.push_back(arguments::mem_obj_arg{hMem, Index, Flags});
  };
};

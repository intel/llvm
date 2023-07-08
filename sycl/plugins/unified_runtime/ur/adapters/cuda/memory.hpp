//===--------- memory.hpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cuda.h>
#include <ur_api.h>

#include "common.hpp"

/// UR Mem mapping to CUDA memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the CUDA backend.
/// Keeps tracks of all mapped regions used for Map/Unmap calls.
/// Only one region can be active at the same time per allocation.
struct ur_mem_handle_t_ {
  // Context where the memory object is accessible
  ur_context_handle_t Context;

  /// Reference counting of the handler
  std::atomic_uint32_t RefCount;
  enum class Type { Buffer, Surface } MemType;

  // Original mem flags passed
  ur_mem_flags_t MemFlags;

  /// A UR Memory object represents either plain memory allocations ("Buffers"
  /// in OpenCL) or typed allocations ("Images" in OpenCL).
  /// In CUDA their API handlers are different. Whereas "Buffers" are allocated
  /// as pointer-like structs, "Images" are stored in Textures or Surfaces.
  /// This union allows implementation to use either from the same handler.
  union MemImpl {
    // Handler for plain, pointer-based CUDA allocations
    struct BufferMem {
      using native_type = CUdeviceptr;

      // If this allocation is a sub-buffer (i.e., a view on an existing
      // allocation), this is the pointer to the parent handler structure
      ur_mem_handle_t Parent;
      // CUDA handler for the pointer
      native_type Ptr;

      /// Pointer associated with this device on the host
      void *HostPtr;
      /// Size of the allocation in bytes
      size_t Size;
      /// Offset of the active mapped region.
      size_t MapOffset;
      /// Pointer to the active mapped region, if any
      void *MapPtr;
      /// Original flags for the mapped region
      ur_map_flags_t MapFlags;

      /** AllocMode
       * classic: Just a normal buffer allocated on the device via cuda malloc
       * use_host_ptr: Use an address on the host for the device
       * copy_in: The data for the device comes from the host but the host
       pointer is not available later for re-use
       * alloc_host_ptr: Uses pinned-memory allocation
      */
      enum class AllocMode {
        Classic,
        UseHostPtr,
        CopyIn,
        AllocHostPtr,
      } MemAllocMode;

      native_type get() const noexcept { return Ptr; }

      size_t getSize() const noexcept { return Size; }

      void *getMapPtr() const noexcept { return MapPtr; }

      size_t getMapOffset(void *) const noexcept { return MapOffset; }

      /// Returns a pointer to data visible on the host that contains
      /// the data on the device associated with this allocation.
      /// The offset is used to index into the CUDA allocation.
      void *mapToPtr(size_t Offset, ur_map_flags_t Flags) noexcept {
        assert(MapPtr == nullptr);
        MapOffset = Offset;
        MapFlags = Flags;
        if (HostPtr) {
          MapPtr = static_cast<char *>(HostPtr) + Offset;
        } else {
          // TODO: Allocate only what is needed based on the offset
          MapPtr = static_cast<void *>(malloc(this->getSize()));
        }
        return MapPtr;
      }

      /// Detach the allocation from the host memory.
      void unmap(void *) noexcept {
        assert(MapPtr != nullptr);

        if (MapPtr != HostPtr) {
          free(MapPtr);
        }
        MapPtr = nullptr;
        MapOffset = 0;
      }

      ur_map_flags_t getMapFlags() const noexcept {
        assert(MapPtr != nullptr);
        return MapFlags;
      }
    } BufferMem;

    // Handler data for surface object (i.e. Images)
    struct SurfaceMem {
      CUarray Array;
      CUsurfObject SurfObj;
      ur_mem_type_t ImageType;

      CUarray getArray() const noexcept { return Array; }

      CUsurfObject getSurface() const noexcept { return SurfObj; }

      ur_mem_type_t getImageType() const noexcept { return ImageType; }
    } SurfaceMem;
  } Mem;

  /// Constructs the UR mem handler for a non-typed allocation ("buffer")
  ur_mem_handle_t_(ur_context_handle_t Context, ur_mem_handle_t Parent,
                   ur_mem_flags_t MemFlags, MemImpl::BufferMem::AllocMode Mode,
                   CUdeviceptr Ptr, void *HostPtr, size_t Size)
      : Context{Context}, RefCount{1}, MemType{Type::Buffer},
        MemFlags{MemFlags} {
    Mem.BufferMem.Ptr = Ptr;
    Mem.BufferMem.Parent = Parent;
    Mem.BufferMem.HostPtr = HostPtr;
    Mem.BufferMem.Size = Size;
    Mem.BufferMem.MapOffset = 0;
    Mem.BufferMem.MapPtr = nullptr;
    Mem.BufferMem.MapFlags = UR_MAP_FLAG_WRITE;
    Mem.BufferMem.MemAllocMode = Mode;
    if (isSubBuffer()) {
      urMemRetain(Mem.BufferMem.Parent);
    } else {
      urContextRetain(Context);
    }
  };

  /// Constructs the UR allocation for an Image object (surface in CUDA)
  ur_mem_handle_t_(ur_context_handle_t Context, CUarray Array,
                   CUsurfObject Surf, ur_mem_flags_t MemFlags,
                   ur_mem_type_t ImageType, void *HostPtr)
      : Context{Context}, RefCount{1}, MemType{Type::Surface},
        MemFlags{MemFlags} {
    (void)HostPtr;

    Mem.SurfaceMem.Array = Array;
    Mem.SurfaceMem.SurfObj = Surf;
    Mem.SurfaceMem.ImageType = ImageType;
    urContextRetain(Context);
  }

  ~ur_mem_handle_t_() {
    if (isBuffer() && isSubBuffer()) {
      urMemRelease(Mem.BufferMem.Parent);
      return;
    }
    urContextRelease(Context);
  }

  bool isBuffer() const noexcept { return MemType == Type::Buffer; }

  bool isSubBuffer() const noexcept {
    return (isBuffer() && (Mem.BufferMem.Parent != nullptr));
  }

  bool isImage() const noexcept { return MemType == Type::Surface; }

  ur_context_handle_t getContext() const noexcept { return Context; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};

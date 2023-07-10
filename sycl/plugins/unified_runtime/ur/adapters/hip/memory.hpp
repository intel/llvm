//===--------- context.cpp - HIP Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include <cassert>

/// UR Mem mapping to HIP memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the HIP backend.
/// Keeps tracks of all mapped regions used for Map/Unmap calls.
/// Only one region can be active at the same time per allocation.
struct ur_mem_handle_t_ {

  // TODO: Move as much shared data up as possible
  using ur_context = ur_context_handle_t_ *;
  using ur_mem = ur_mem_handle_t_ *;

  // Context where the memory object is accessible
  ur_context Context;

  /// Reference counting of the handler
  std::atomic_uint32_t RefCount;
  enum class Type { Buffer, Surface } MemType;

  // Original mem flags passed
  ur_mem_flags_t MemFlags;

  /// A UR Memory object represents either plain memory allocations ("Buffers"
  /// in OpenCL) or typed allocations ("Images" in OpenCL).
  /// In HIP their API handlers are different. Whereas "Buffers" are allocated
  /// as pointer-like structs, "Images" are stored in Textures or Surfaces.
  /// This union allows implementation to use either from the same handler.
  union MemImpl {
    // Handler for plain, pointer-based HIP allocations
    struct BufferMem {
      using native_type = hipDeviceptr_t;

      // If this allocation is a sub-buffer (i.e., a view on an existing
      // allocation), this is the pointer to the parent handler structure
      ur_mem Parent;
      // HIP handler for the pointer
      native_type Ptr;

      /// Pointer associated with this device on the host
      void *HostPtr;
      /// Size of the allocation in bytes
      size_t Size;
      /// Size of the active mapped region.
      size_t MapSize;
      /// Offset of the active mapped region.
      size_t MapOffset;
      /// Pointer to the active mapped region, if any
      void *MapPtr;
      /// Original flags for the mapped region
      ur_map_flags_t MapFlags;

      /** AllocMode
       * Classic: Just a normal buffer allocated on the device via hip malloc
       * UseHostPtr: Use an address on the host for the device
       * CopyIn: The data for the device comes from the host but the host
       pointer is not available later for re-use
       * AllocHostPtr: Uses pinned-memory allocation
      */
      enum class AllocMode {
        Classic,
        UseHostPtr,
        CopyIn,
        AllocHostPtr
      } MemAllocMode;

      native_type get() const noexcept { return Ptr; }

      native_type getWithOffset(size_t Offset) const noexcept {
        return reinterpret_cast<native_type>(reinterpret_cast<uint8_t *>(Ptr) +
                                             Offset);
      }

      void *getVoid() const noexcept { return reinterpret_cast<void *>(Ptr); }

      size_t getSize() const noexcept { return Size; }

      void *getMapPtr() const noexcept { return MapPtr; }

      size_t getMapSize() const noexcept { return MapSize; }

      size_t getMapOffset() const noexcept { return MapOffset; }

      /// Returns a pointer to data visible on the host that contains
      /// the data on the device associated with this allocation.
      /// The offset is used to index into the HIP allocation.
      ///
      void *mapToPtr(size_t Size, size_t Offset,
                     ur_map_flags_t Flags) noexcept {
        assert(MapPtr == nullptr);
        MapSize = Size;
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
        MapSize = 0;
        MapOffset = 0;
      }

      ur_map_flags_t getMapFlags() const noexcept {
        assert(MapPtr != nullptr);
        return MapFlags;
      }
    } BufferMem;

    // Handler data for surface object (i.e. Images)
    struct SurfaceMem {
      hipArray *Array;
      hipSurfaceObject_t SurfObj;
      ur_mem_type_t ImageType;

      hipArray *getArray() const noexcept { return Array; }

      hipSurfaceObject_t getSurface() const noexcept { return SurfObj; }

      ur_mem_type_t getImageType() const noexcept { return ImageType; }
    } SurfaceMem;
  } Mem;

  /// Constructs the UR MEM handler for a non-typed allocation ("buffer")
  ur_mem_handle_t_(ur_context Ctxt, ur_mem Parent, ur_mem_flags_t MemFlags,
                   MemImpl::BufferMem::AllocMode Mode, hipDeviceptr_t Ptr,
                   void *HostPtr, size_t Size)
      : Context{Ctxt}, RefCount{1}, MemType{Type::Buffer}, MemFlags{MemFlags} {
    Mem.BufferMem.Ptr = Ptr;
    Mem.BufferMem.Parent = Parent;
    Mem.BufferMem.HostPtr = HostPtr;
    Mem.BufferMem.Size = Size;
    Mem.BufferMem.MapSize = 0;
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

  /// Constructs the UR allocation for an Image object
  ur_mem_handle_t_(ur_context Ctxt, hipArray *Array, hipSurfaceObject_t Surf,
                   ur_mem_flags_t MemFlags, ur_mem_type_t ImageType, void *)
      : Context{Ctxt}, RefCount{1}, MemType{Type::Surface}, MemFlags{MemFlags} {
    Mem.SurfaceMem.Array = Array;
    Mem.SurfaceMem.ImageType = ImageType;
    Mem.SurfaceMem.SurfObj = Surf;
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

  ur_context getContext() const noexcept { return Context; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};

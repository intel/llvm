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

  // Context where the memory object is accessibles
  ur_context context_;

  /// Reference counting of the handler
  std::atomic_uint32_t refCount_;
  enum class mem_type { buffer, surface } mem_type_;

  // Original mem flags passed
  ur_mem_flags_t memFlags_;

  /// A UR Memory object represents either plain memory allocations ("Buffers"
  /// in OpenCL) or typed allocations ("Images" in OpenCL).
  /// In HIP their API handlers are different. Whereas "Buffers" are allocated
  /// as pointer-like structs, "Images" are stored in Textures or Surfaces
  /// This union allows implementation to use either from the same handler.
  union mem_ {
    // Handler for plain, pointer-based HIP allocations
    struct buffer_mem_ {
      using native_type = hipDeviceptr_t;

      // If this allocation is a sub-buffer (i.e., a view on an existing
      // allocation), this is the pointer to the parent handler structure
      ur_mem parent_;
      // HIP handler for the pointer
      native_type ptr_;

      /// Pointer associated with this device on the host
      void *hostPtr_;
      /// Size of the allocation in bytes
      size_t size_;
      /// Offset of the active mapped region.
      size_t mapOffset_;
      /// Pointer to the active mapped region, if any
      void *mapPtr_;
      /// Original flags for the mapped region
      ur_map_flags_t mapFlags_;

      /** alloc_mode
       * classic: Just a normal buffer allocated on the device via hip malloc
       * use_host_ptr: Use an address on the host for the device
       * copy_in: The data for the device comes from the host but the host
       pointer is not available later for re-use
       * alloc_host_ptr: Uses pinned-memory allocation
      */
      enum class alloc_mode {
        classic,
        use_host_ptr,
        copy_in,
        alloc_host_ptr
      } allocMode_;

      native_type get() const noexcept { return ptr_; }

      native_type get_with_offset(size_t offset) const noexcept {
        return reinterpret_cast<native_type>(reinterpret_cast<uint8_t *>(ptr_) +
                                             offset);
      }

      void *get_void() const noexcept { return reinterpret_cast<void *>(ptr_); }

      size_t get_size() const noexcept { return size_; }

      void *get_map_ptr() const noexcept { return mapPtr_; }

      size_t get_map_offset(void *ptr) const noexcept {
        (void)ptr;
        return mapOffset_;
      }

      /// Returns a pointer to data visible on the host that contains
      /// the data on the device associated with this allocation.
      /// The offset is used to index into the HIP allocation.
      ///
      void *map_to_ptr(size_t offset, ur_map_flags_t flags) noexcept {
        assert(mapPtr_ == nullptr);
        mapOffset_ = offset;
        mapFlags_ = flags;
        if (hostPtr_) {
          mapPtr_ = static_cast<char *>(hostPtr_) + offset;
        } else {
          // TODO: Allocate only what is needed based on the offset
          mapPtr_ = static_cast<void *>(malloc(this->get_size()));
        }
        return mapPtr_;
      }

      /// Detach the allocation from the host memory.
      void unmap(void *ptr) noexcept {
        (void)ptr;
        assert(mapPtr_ != nullptr);

        if (mapPtr_ != hostPtr_) {
          free(mapPtr_);
        }
        mapPtr_ = nullptr;
        mapOffset_ = 0;
      }

      ur_map_flags_t get_map_flags() const noexcept {
        assert(mapPtr_ != nullptr);
        return mapFlags_;
      }
    } buffer_mem_;

    // Handler data for surface object (i.e. Images)
    struct surface_mem_ {
      hipArray *array_;
      hipSurfaceObject_t surfObj_;
      ur_mem_type_t imageType_;

      hipArray *get_array() const noexcept { return array_; }

      hipSurfaceObject_t get_surface() const noexcept { return surfObj_; }

      ur_mem_type_t get_image_type() const noexcept { return imageType_; }
    } surface_mem_;
  } mem_;

  /// Constructs the UR MEM handler for a non-typed allocation ("buffer")
  ur_mem_handle_t_(ur_context ctxt, ur_mem parent, ur_mem_flags_t mem_flags,
                   mem_::buffer_mem_::alloc_mode mode, hipDeviceptr_t ptr,
                   void *host_ptr, size_t size)
      : context_{ctxt}, refCount_{1}, mem_type_{mem_type::buffer},
        memFlags_{mem_flags} {
    mem_.buffer_mem_.ptr_ = ptr;
    mem_.buffer_mem_.parent_ = parent;
    mem_.buffer_mem_.hostPtr_ = host_ptr;
    mem_.buffer_mem_.size_ = size;
    mem_.buffer_mem_.mapOffset_ = 0;
    mem_.buffer_mem_.mapPtr_ = nullptr;
    mem_.buffer_mem_.mapFlags_ = UR_MAP_FLAG_WRITE;
    mem_.buffer_mem_.allocMode_ = mode;
    if (is_sub_buffer()) {
      urMemRetain(mem_.buffer_mem_.parent_);
    } else {
      urContextRetain(context_);
    }
  };

  /// Constructs the UR allocation for an Image object
  ur_mem_handle_t_(ur_context ctxt, hipArray *array, hipSurfaceObject_t surf,
                   ur_mem_flags_t mem_flags, ur_mem_type_t image_type,
                   void *host_ptr)
      : context_{ctxt}, refCount_{1}, mem_type_{mem_type::surface},
        memFlags_{mem_flags} {
    (void)host_ptr;
    mem_.surface_mem_.array_ = array;
    mem_.surface_mem_.imageType_ = image_type;
    mem_.surface_mem_.surfObj_ = surf;
    urContextRetain(context_);
  }

  ~ur_mem_handle_t_() {
    if (mem_type_ == mem_type::buffer) {
      if (is_sub_buffer()) {
        urMemRelease(mem_.buffer_mem_.parent_);
        return;
      }
    }
    urContextRelease(context_);
  }

  // TODO: Move as many shared funcs up as possible
  bool is_buffer() const noexcept { return mem_type_ == mem_type::buffer; }

  bool is_sub_buffer() const noexcept {
    return (is_buffer() && (mem_.buffer_mem_.parent_ != nullptr));
  }

  bool is_image() const noexcept { return mem_type_ == mem_type::surface; }

  ur_context get_context() const noexcept { return context_; }

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }
};

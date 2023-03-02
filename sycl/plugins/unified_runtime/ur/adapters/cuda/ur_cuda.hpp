//===--------- ur_cuda.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

// We need the PI header temporarily while the UR device impl still uses the
// PI context type
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

#include <cuda.h>

struct ur_platform_handle_t_ : public _ur_platform {
  std::vector<std::unique_ptr<ur_device_handle_t_>> devices_;
};

struct ur_device_handle_t_ : public _pi_object {
private:
  using native_type = CUdevice;

  native_type cuDevice_;
  CUcontext cuContext_;
  CUevent evBase_; // CUDA event used as base counter
  std::atomic_uint32_t refCount_;
  ur_platform_handle_t platform_;

  static constexpr pi_uint32 max_work_item_dimensions = 3u;
  size_t max_work_item_sizes[max_work_item_dimensions];
  int max_work_group_size;

public:
  ur_device_handle_t_(native_type cuDevice, CUcontext cuContext, CUevent evBase,
                      ur_platform_handle_t platform)
      : cuDevice_(cuDevice), cuContext_(cuContext),
        evBase_(evBase), refCount_{1}, platform_(platform) {}

  ur_device_handle_t_() { cuDevicePrimaryCtxRelease(cuDevice_); }

  native_type get() const noexcept { return cuDevice_; };

  CUcontext get_context() const noexcept { return cuContext_; };

  uint32_t get_reference_count() const noexcept { return refCount_; }

  ur_platform_handle_t get_platform() const noexcept { return platform_; };

  uint64_t get_elapsed_time(CUevent) const;

  void save_max_work_item_sizes(size_t size,
                                size_t *save_max_work_item_sizes) noexcept {
    memcpy(max_work_item_sizes, save_max_work_item_sizes, size);
  };

  void save_max_work_group_size(int value) noexcept {
    max_work_group_size = value;
  };

  void get_max_work_item_sizes(size_t ret_size,
                               size_t *ret_max_work_item_sizes) const noexcept {
    memcpy(ret_max_work_item_sizes, max_work_item_sizes, ret_size);
  };

  int get_max_work_group_size() const noexcept { return max_work_group_size; };
};

struct ur_context_handle_t_ : _pi_object {

  struct deleter_data {
    pi_context_extended_deleter function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  using native_type = CUcontext;

  native_type cuContext_;
  ur_device_handle_t deviceId_;
  std::atomic_uint32_t refCount_;

  ur_context_handle_t_(ur_device_handle_t_ *devId)
      : cuContext_{devId->get_context()}, deviceId_{devId}, refCount_{1} {
    urDeviceRetain(deviceId_);
  };

  ~ur_context_handle_t_() { urDeviceRelease(deviceId_); }

  void invoke_extended_deleters() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto &deleter : extended_deleters_) {
      deleter();
    }
  }

  void set_extended_deleter(pi_context_extended_deleter function,
                            void *user_data) {
    std::lock_guard<std::mutex> guard(mutex_);
    extended_deleters_.emplace_back(deleter_data{function, user_data});
  }

  ur_device_handle_t get_device() const noexcept { return deviceId_; }

  native_type get() const noexcept { return cuContext_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
};

/// UR Mem mapping to CUDA memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the CUDA backend.
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

  /// A UR Memory object represents either plain memory allocations ("Buffers"
  /// in OpenCL) or typed allocations ("Images" in OpenCL).
  /// In CUDA their API handlers are different. Whereas "Buffers" are allocated
  /// as pointer-like structs, "Images" are stored in Textures or Surfaces
  /// This union allows implementation to use either from the same handler.
  union mem_ {
    // Handler for plain, pointer-based CUDA allocations
    struct buffer_mem_ {
      using native_type = CUdeviceptr;

      // If this allocation is a sub-buffer (i.e., a view on an existing
      // allocation), this is the pointer to the parent handler structure
      ur_mem parent_;
      // CUDA handler for the pointer
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
      ur_map_flag_t mapFlags_;

      /** alloc_mode
       * classic: Just a normal buffer allocated on the device via cuda malloc
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

      size_t get_size() const noexcept { return size_; }

      void *get_map_ptr() const noexcept { return mapPtr_; }

      size_t get_map_offset(void *) const noexcept { return mapOffset_; }

      /// Returns a pointer to data visible on the host that contains
      /// the data on the device associated with this allocation.
      /// The offset is used to index into the CUDA allocation.
      ///
      void *map_to_ptr(size_t offset, ur_map_flag_t flags) noexcept {
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
      void unmap(void *) noexcept {
        assert(mapPtr_ != nullptr);

        if (mapPtr_ != hostPtr_) {
          free(mapPtr_);
        }
        mapPtr_ = nullptr;
        mapOffset_ = 0;
      }

      ur_map_flag_t get_map_flags() const noexcept {
        assert(mapPtr_ != nullptr);
        return mapFlags_;
      }
    } buffer_mem_;

    // Handler data for surface object (i.e. Images)
    struct surface_mem_ {
      CUarray array_;
      CUsurfObject surfObj_;
      ur_mem_type_t imageType_;

      CUarray get_array() const noexcept { return array_; }

      CUsurfObject get_surface() const noexcept { return surfObj_; }

      ur_mem_type_t get_image_type() const noexcept { return imageType_; }
    } surface_mem_;
  } mem_;

  /// Constructs the UR mem handler for a non-typed allocation ("buffer")
  ur_mem_handle_t_(ur_context ctxt, ur_mem parent,
                   mem_::buffer_mem_::alloc_mode mode, CUdeviceptr ptr,
                   void *host_ptr, size_t size)
      : context_{ctxt}, refCount_{1}, mem_type_{mem_type::buffer} {
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

  /// Constructs the UR allocation for an Image object (surface in CUDA)
  ur_mem_handle_t_(ur_context ctxt, CUarray array, CUsurfObject surf,
                   ur_mem_type_t image_type, void *host_ptr)
      : context_{ctxt}, refCount_{1}, mem_type_{mem_type::surface} {
    // Ignore unused parameter
    (void)host_ptr;

    mem_.surface_mem_.array_ = array;
    mem_.surface_mem_.surfObj_ = surf;
    mem_.surface_mem_.imageType_ = image_type;
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

// Make the Unified Runtime handles definition complete.
// This is used in various "create" API where new handles are allocated.
// struct _zer_platform_handle_t : public _ur_platform_handle_t {
//   using _ur_platform_handle_t::_ur_platform_handle_t;
// };
// struct _zer_device_handle_t : public _ur_device_handle_t {
//   using _ur_device_handle_t::_ur_device_handle_t;
// };

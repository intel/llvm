//===-- pi_cuda.hpp - CUDA Plugin -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_cuda CUDA Plugin
/// \ingroup sycl_pi

/// \file pi_cuda.hpp
/// Declarations for CUDA Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CUDA runtime.
///
/// \ingroup sycl_pi_cuda

#ifndef PI_CUDA_HPP
#define PI_CUDA_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_CUDA_PLUGIN_VERSION 1

#define _PI_CUDA_PLUGIN_VERSION_STRING                                         \
  _PI_PLUGIN_VERSION_STRING(_PI_CUDA_PLUGIN_VERSION)

#include "sycl/detail/pi.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/adapters/cuda/ur_cuda.hpp>

extern "C" {

/// \cond IGNORE_BLOCK_IN_DOXYGEN
pi_result cuda_piContextRetain(pi_context);
pi_result cuda_piContextRelease(pi_context);
pi_result cuda_piDeviceRelease(pi_device);
pi_result cuda_piDeviceRetain(pi_device);
pi_result cuda_piProgramRetain(pi_program);
pi_result cuda_piProgramRelease(pi_program);
pi_result cuda_piQueueRelease(pi_queue);
pi_result cuda_piQueueRetain(pi_queue);
pi_result cuda_piMemRetain(pi_mem);
pi_result cuda_piMemRelease(pi_mem);
pi_result cuda_piKernelRetain(pi_kernel);
pi_result cuda_piKernelRelease(pi_kernel);
pi_result cuda_piKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                    pi_kernel_group_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret);
/// \endcond
}

using _pi_stream_guard = std::unique_lock<std::mutex>;

/// A PI platform stores all known PI devices,
///  in the CUDA plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct _pi_platform : ur_platform_handle_t_ {
  using ur_platform_handle_t_::ur_platform_handle_t_;
};

/// PI device mapping to a CUdevice.
/// Includes an observer pointer to the platform,
/// and implements the reference counting semantics since
/// CUDA objects are not refcounted.
///
struct _pi_device : ur_device_handle_t_ {
  using ur_device_handle_t_::ur_device_handle_t_;
};

/// PI context mapping to a CUDA context object.
///
/// There is no direct mapping between a CUDA context and a PI context,
/// main differences described below:
///
/// <b> CUDA context vs PI context </b>
///
/// One of the main differences between the PI API and the CUDA driver API is
/// that the second modifies the state of the threads by assigning
/// `CUcontext` objects to threads. `CUcontext` objects store data associated
/// with a given device and control access to said device from the user side.
/// PI API context are objects that are passed to functions, and not bound
/// to threads.
/// The _pi_context object doesn't implement this behavior, only holds the
/// CUDA context data. The RAII object \ref ScopedContext implements the active
/// context behavior.
///
/// <b> Primary vs User-defined context </b>
///
/// CUDA has two different types of context, the Primary context,
/// which is usable by all threads on a given process for a given device, and
/// the aforementioned custom contexts.
/// CUDA documentation, and performance analysis, indicates it is recommended
/// to use Primary context whenever possible.
/// Primary context is used as well by the CUDA Runtime API.
/// For PI applications to interop with CUDA Runtime API, they have to use
/// the primary context - and make that active in the thread.
/// The `_pi_context` object can be constructed with a `kind` parameter
/// that allows to construct a Primary or `user-defined` context, so that
/// the PI object interface is always the same.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the PI Context can store a number of callback functions that will be
///  called upon destruction of the PI Context.
///  See proposal for details.
///
struct _pi_context : ur_context_handle_t_ {
  using ur_context_handle_t_::ur_context_handle_t_;
};

/// PI Mem mapping to CUDA memory allocations, both data and texture/surface.
/// \brief Represents non-SVM allocations on the CUDA backend.
/// Keeps tracks of all mapped regions used for Map/Unmap calls.
/// Only one region can be active at the same time per allocation.
struct _pi_mem {

  // TODO: Move as much shared data up as possible
  using pi_context = _pi_context *;

  // Context where the memory object is accessibles
  pi_context context_;

  /// Reference counting of the handler
  std::atomic_uint32_t refCount_;
  enum class mem_type { buffer, surface } mem_type_;

  /// A PI Memory object represents either plain memory allocations ("Buffers"
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
      pi_mem parent_;
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
      pi_map_flags mapFlags_;

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
      void *map_to_ptr(size_t offset, pi_map_flags flags) noexcept {
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

      pi_map_flags get_map_flags() const noexcept {
        assert(mapPtr_ != nullptr);
        return mapFlags_;
      }
    } buffer_mem_;

    // Handler data for surface object (i.e. Images)
    struct surface_mem_ {
      CUarray array_;
      CUsurfObject surfObj_;
      pi_mem_type imageType_;

      CUarray get_array() const noexcept { return array_; }

      CUsurfObject get_surface() const noexcept { return surfObj_; }

      pi_mem_type get_image_type() const noexcept { return imageType_; }
    } surface_mem_;
  } mem_;

  /// Constructs the PI MEM handler for a non-typed allocation ("buffer")
  _pi_mem(pi_context ctxt, pi_mem parent, mem_::buffer_mem_::alloc_mode mode,
          CUdeviceptr ptr, void *host_ptr, size_t size)
      : context_{ctxt}, refCount_{1}, mem_type_{mem_type::buffer} {
    mem_.buffer_mem_.ptr_ = ptr;
    mem_.buffer_mem_.parent_ = parent;
    mem_.buffer_mem_.hostPtr_ = host_ptr;
    mem_.buffer_mem_.size_ = size;
    mem_.buffer_mem_.mapOffset_ = 0;
    mem_.buffer_mem_.mapPtr_ = nullptr;
    mem_.buffer_mem_.mapFlags_ = PI_MAP_WRITE;
    mem_.buffer_mem_.allocMode_ = mode;
    if (is_sub_buffer()) {
      cuda_piMemRetain(mem_.buffer_mem_.parent_);
    } else {
      cuda_piContextRetain(context_);
    }
  };

  /// Constructs the PI allocation for an Image object (surface in CUDA)
  _pi_mem(pi_context ctxt, CUarray array, CUsurfObject surf,
          pi_mem_type image_type, void *host_ptr)
      : context_{ctxt}, refCount_{1}, mem_type_{mem_type::surface} {
    // Ignore unused parameter
    (void)host_ptr;

    mem_.surface_mem_.array_ = array;
    mem_.surface_mem_.surfObj_ = surf;
    mem_.surface_mem_.imageType_ = image_type;
    cuda_piContextRetain(context_);
  }

  ~_pi_mem() {
    if (mem_type_ == mem_type::buffer) {
      if (is_sub_buffer()) {
        cuda_piMemRelease(mem_.buffer_mem_.parent_);
        return;
      }
    }
    cuda_piContextRelease(context_);
  }

  // TODO: Move as many shared funcs up as possible
  bool is_buffer() const noexcept { return mem_type_ == mem_type::buffer; }

  bool is_sub_buffer() const noexcept {
    return (is_buffer() && (mem_.buffer_mem_.parent_ != nullptr));
  }

  bool is_image() const noexcept { return mem_type_ == mem_type::surface; }

  pi_context get_context() const noexcept { return context_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

/// PI queue mapping on to CUstream objects.
///
struct _pi_queue : ur_queue_handle_t_ {
  using ur_queue_handle_t_::ur_queue_handle_t_;
};

typedef void (*pfn_notify)(pi_event event, pi_int32 eventCommandStatus,
                           void *userData);

struct _pi_event : ur_event_handle_t_ {
  using ur_event_handle_t_::ur_event_handle_t_;

  // Helpers for queue command implementations until they also get ported to UR
  static pi_event
  make_native(pi_command_type type, pi_queue queue, CUstream stream,
              uint32_t stream_token = std::numeric_limits<uint32_t>::max()) {
    auto urQueue = reinterpret_cast<ur_queue_handle_t>(queue);
    static std::unordered_map<_pi_command_type, ur_command_t> cmdMap = {
        {PI_COMMAND_TYPE_NDRANGE_KERNEL, UR_COMMAND_KERNEL_LAUNCH},
        {PI_COMMAND_TYPE_MEM_BUFFER_READ, UR_COMMAND_MEM_BUFFER_READ},
        {PI_COMMAND_TYPE_MEM_BUFFER_WRITE, UR_COMMAND_MEM_BUFFER_WRITE},
        {PI_COMMAND_TYPE_MEM_BUFFER_COPY, UR_COMMAND_MEM_BUFFER_COPY},
        {PI_COMMAND_TYPE_MEM_BUFFER_MAP, UR_COMMAND_MEM_BUFFER_MAP},
        {PI_COMMAND_TYPE_MEM_BUFFER_UNMAP, UR_COMMAND_MEM_UNMAP},
        {PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT, UR_COMMAND_MEM_BUFFER_READ_RECT},
        {PI_COMMAND_TYPE_MEM_BUFFER_WRITE_RECT,
         UR_COMMAND_MEM_BUFFER_WRITE_RECT},
        {PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, UR_COMMAND_MEM_BUFFER_COPY_RECT},
        {PI_COMMAND_TYPE_MEM_BUFFER_FILL, UR_COMMAND_MEM_BUFFER_FILL},
        {PI_COMMAND_TYPE_IMAGE_READ, UR_COMMAND_MEM_IMAGE_READ},
        {PI_COMMAND_TYPE_IMAGE_WRITE, UR_COMMAND_MEM_IMAGE_WRITE},
        {PI_COMMAND_TYPE_IMAGE_COPY, UR_COMMAND_MEM_IMAGE_COPY},
        {PI_COMMAND_TYPE_BARRIER, UR_COMMAND_EVENTS_WAIT_WITH_BARRIER},
        {PI_COMMAND_TYPE_DEVICE_GLOBAL_VARIABLE_READ,
         UR_COMMAND_DEVICE_GLOBAL_VARIABLE_READ},
        {PI_COMMAND_TYPE_DEVICE_GLOBAL_VARIABLE_WRITE,
         UR_COMMAND_DEVICE_GLOBAL_VARIABLE_WRITE},
    };

    // TODO(ur): There is no exact mapping for the following commands. Just
    // default to KERNEL_LAUNCH for now.
    // PI_COMMAND_TYPE_USER
    // PI_COMMAND_TYPE_MEM_BUFFER_FILL,
    // PI_COMMAND_TYPE_IMAGE_READ,
    // PI_COMMAND_TYPE_IMAGE_WRITE,
    // PI_COMMAND_TYPE_IMAGE_COPY,
    // PI_COMMAND_TYPE_NATIVE_KERNEL,
    // PI_COMMAND_TYPE_COPY_BUFFER_TO_IMAGE,
    // PI_COMMAND_TYPE_COPY_IMAGE_TO_BUFFER,
    // PI_COMMAND_TYPE_MAP_IMAGE,
    // PI_COMMAND_TYPE_MARKER,
    // PI_COMMAND_TYPE_ACQUIRE_GL_OBJECTS,
    // PI_COMMAND_TYPE_RELEASE_GL_OBJECTS,
    // PI_COMMAND_TYPE_BARRIER,
    // PI_COMMAND_TYPE_MIGRATE_MEM_OBJECTS,
    // PI_COMMAND_TYPE_FILL_IMAGE
    // PI_COMMAND_TYPE_SVM_FREE
    // PI_COMMAND_TYPE_SVM_MEMCPY
    // PI_COMMAND_TYPE_SVM_MEMFILL
    // PI_COMMAND_TYPE_SVM_MAP
    // PI_COMMAND_TYPE_SVM_UNMAP

    ur_command_t urCmd = UR_COMMAND_KERNEL_LAUNCH;
    auto cmdIt = cmdMap.find(type);
    if (cmdIt != cmdMap.end()) {
      urCmd = cmdIt->second;
    }
    return reinterpret_cast<pi_event>(
        ur_event_handle_t_::make_native(urCmd, urQueue, stream, stream_token));
  }

  static pi_event make_with_native(ur_context_handle_t context,
                                   CUevent eventNative) {
    auto urContext = reinterpret_cast<ur_context_handle_t>(context);
    return reinterpret_cast<pi_event>(
        ur_event_handle_t_::make_with_native(urContext, eventNative));
  }
};

/// Implementation of PI Program on CUDA Module object
///
struct _pi_program {
  using native_type = CUmodule;
  native_type module_;
  const char *binary_;
  size_t binarySizeInBytes_;
  std::atomic_uint32_t refCount_;
  _pi_context *context_;

  // Metadata
  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>>
      kernelReqdWorkGroupSizeMD_;
  std::unordered_map<std::string, std::string> globalIDMD_;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char errorLog_[MAX_LOG_SIZE], infoLog_[MAX_LOG_SIZE];
  std::string buildOptions_;
  pi_program_build_status buildStatus_ = PI_PROGRAM_BUILD_STATUS_NONE;

  _pi_program(pi_context ctxt);
  ~_pi_program();

  pi_result set_metadata(const pi_device_binary_property *metadata,
                         size_t length);

  pi_result set_binary(const char *binary, size_t binarySizeInBytes);

  pi_result build_program(const char *build_options);

  pi_context get_context() const { return context_; };

  native_type get() const noexcept { return module_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

/// Implementation of a PI Kernel for CUDA
///
/// PI Kernels are used to set kernel arguments,
/// creating a state on the Kernel object for a given
/// invocation. This is not the case of CUFunction objects,
/// which are simply passed together with the arguments on the invocation.
/// The PI Kernel implementation for CUDA stores the list of arguments,
/// argument sizes and offsets to emulate the interface of PI Kernel,
/// saving the arguments for the later dispatch.
/// Note that in PI API, the Local memory is specified as a size per
/// individual argument, but in CUDA only the total usage of shared
/// memory is required since it is not passed as a parameter.
/// A compiler pass converts the PI API local memory model into the
/// CUDA shared model. This object simply calculates the total of
/// shared memory, and the initial offsets of each parameter.
///
struct _pi_kernel {
  using native_type = CUfunction;

  native_type function_;
  native_type functionWithOffsetParam_;
  std::string name_;
  pi_context context_;
  pi_program program_;
  std::atomic_uint32_t refCount_;

  static constexpr pi_uint32 REQD_THREADS_PER_BLOCK_DIMENSIONS = 3u;
  size_t reqdThreadsPerBlock_[REQD_THREADS_PER_BLOCK_DIMENSIONS];

  /// Structure that holds the arguments to the kernel.
  /// Note earch argument size is known, since it comes
  /// from the kernel signature.
  /// This is not something can be queried from the CUDA API
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

    pi_uint32 get_local_size() const {
      return std::accumulate(std::begin(offsetPerIndex_),
                             std::end(offsetPerIndex_), 0);
    }
  } args_;

  _pi_kernel(CUfunction func, CUfunction funcWithOffsetParam, const char *name,
             pi_program program, pi_context ctxt)
      : function_{func}, functionWithOffsetParam_{funcWithOffsetParam},
        name_{name}, context_{ctxt}, program_{program}, refCount_{1} {
    cuda_piProgramRetain(program_);
    cuda_piContextRetain(context_);
    /// Note: this code assumes that there is only one device per context
    pi_result retError = cuda_piKernelGetGroupInfo(
        this, reinterpret_cast<pi_device>(ctxt->get_device()),
        PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(reqdThreadsPerBlock_), reqdThreadsPerBlock_, nullptr);
    (void)retError;
    assert(retError == PI_SUCCESS);
  }

  ~_pi_kernel() {
    cuda_piProgramRelease(program_);
    cuda_piContextRelease(context_);
  }

  pi_program get_program() const noexcept { return program_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  native_type get() const noexcept { return function_; };

  native_type get_with_offset_parameter() const noexcept {
    return functionWithOffsetParam_;
  };

  bool has_with_offset_parameter() const noexcept {
    return functionWithOffsetParam_ != nullptr;
  }

  pi_context get_context() const noexcept { return context_; };

  const char *get_name() const noexcept { return name_.c_str(); }

  /// Returns the number of arguments, excluding the implicit global offset.
  /// Note this only returns the current known number of arguments, not the
  /// real one required by the kernel, since this cannot be queried from
  /// the CUDA Driver API
  pi_uint32 get_num_args() const noexcept { return args_.indices_.size() - 1; }

  void set_kernel_arg(int index, size_t size, const void *arg) {
    args_.add_arg(index, size, arg);
  }

  void set_kernel_local_arg(int index, size_t size) {
    args_.add_local_arg(index, size);
  }

  void set_implicit_offset_arg(size_t size, std::uint32_t *implicitOffset) {
    args_.set_implicit_offset(size, implicitOffset);
  }

  const arguments::args_index_t &get_arg_indices() const {
    return args_.get_indices();
  }

  pi_uint32 get_local_size() const noexcept { return args_.get_local_size(); }

  void clear_local_size() { args_.clear_local_size(); }
};

/// Implementation of samplers for CUDA
///
/// Sampler property layout:
/// | 31 30 ... 6 5 |      4 3 2      |     1      |         0        |
/// |      N/A      | addressing mode | fiter mode | normalize coords |
struct _pi_sampler {
  std::atomic_uint32_t refCount_;
  pi_uint32 props_;
  pi_context context_;

  _pi_sampler(pi_context context)
      : refCount_(1), props_(0), context_(context) {}

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

// -------------------------------------------------------------
// Helper types and functions
//

#endif // PI_CUDA_HPP

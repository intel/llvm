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

#include "CL/sycl/detail/pi.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <limits>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>
#include <functional>
#include <mutex>

extern "C" {

/// \cond INGORE_BLOCK_IN_DOXYGEN
pi_result cuda_piContextRetain(pi_context );
pi_result cuda_piContextRelease(pi_context );
pi_result cuda_piDeviceRelease(pi_device );
pi_result cuda_piDeviceRetain(pi_device );
pi_result cuda_piProgramRetain(pi_program );
pi_result cuda_piProgramRelease(pi_program );
pi_result cuda_piQueueRelease(pi_queue);
pi_result cuda_piQueueRetain(pi_queue);
pi_result cuda_piMemRetain(pi_mem);
pi_result cuda_piMemRelease(pi_mem);
pi_result cuda_piKernelRetain(pi_kernel);
pi_result cuda_piKernelRelease(pi_kernel);
/// \endcond
}

/// A PI platform stores all known PI devices,
///  in the CUDA plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct _pi_platform {
  std::vector<std::unique_ptr<_pi_device>> devices_;
};

/// PI device mapping to a CUdevice.
/// Includes an observer pointer to the platform,
/// and implements the reference counting semantics since
/// CUDA objects are not refcounted.
///
class _pi_device {
  using native_type = CUdevice;

  native_type cuDevice_;
  std::atomic_uint32_t refCount_;
  pi_platform platform_;

public:
  _pi_device(native_type cuDevice, pi_platform platform)
      : cuDevice_(cuDevice), refCount_{1}, platform_(platform) {}

  native_type get() const noexcept { return cuDevice_; };

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  pi_platform get_platform() const noexcept { return platform_; };
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
struct _pi_context {

  struct deleter_data {
    pi_context_extended_deleter function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  using native_type = CUcontext;

  enum class kind { primary, user_defined } kind_;
  native_type cuContext_;
  _pi_device *deviceId_;
  std::atomic_uint32_t refCount_;

  CUevent evBase_; // CUDA event used as base counter

  _pi_context(kind k, CUcontext ctxt, _pi_device *devId)
      : kind_{k}, cuContext_{ctxt}, deviceId_{devId}, refCount_{1},
        evBase_(nullptr) {
    cuda_piDeviceRetain(deviceId_);
  };

  ~_pi_context() { cuda_piDeviceRelease(deviceId_); }

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

  pi_device get_device() const noexcept { return deviceId_; }

  native_type get() const noexcept { return cuContext_; }

  bool is_primary() const noexcept { return kind_ == kind::primary; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
};

/// PI Mem mapping to a CUDA memory allocation
///
struct _pi_mem {
  using native_type = CUdeviceptr;
  using pi_context = _pi_context *;

  pi_context context_;
  pi_mem parent_;
  native_type ptr_;

  void *hostPtr_;
  size_t size_;
  size_t mapOffset_;
  void *mapPtr_;
  cl_map_flags mapFlags_;
  std::atomic_uint32_t refCount_;
  /** alloc_mode
   * classic: Just a normal buffer allocated on the device via cuda malloc
   * use_host_ptr: Use an address on the host for the device
   * copy_in: The data for the device comes from the host but the host pointer
      is not available later for re-use
  */
  enum class alloc_mode { classic, use_host_ptr, copy_in } allocMode_;

  _pi_mem(pi_context ctxt, pi_mem parent, alloc_mode mode, CUdeviceptr ptr, void *host_ptr,
          size_t size)
      : context_{ctxt}, parent_{parent}, ptr_{ptr}, hostPtr_{host_ptr}, size_{size}, 
        mapOffset_{0}, mapPtr_{nullptr}, mapFlags_{CL_MAP_WRITE}, refCount_{1}, allocMode_{mode} {
      if (is_sub_buffer()) {
        cuda_piMemRetain(parent_);
      } else {
	      cuda_piContextRetain(context_);
      }
	};

   ~_pi_mem() { 
     if (is_sub_buffer()) {
       cuda_piMemRelease(parent_);
     } else {
      cuda_piContextRelease(context_); 
     }
   }

   /// \TODO: Adapt once images are supported.
   bool is_buffer() const noexcept { return true; }

   bool is_sub_buffer() const noexcept {
     return (is_buffer() && (parent_ != nullptr));
   }

  native_type get() const noexcept { return ptr_; }

  pi_context get_context() const noexcept { return context_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  size_t get_size() const noexcept { return size_; }

  void *get_map_ptr() const noexcept { return mapPtr_; }

  size_t get_map_offset(void *ptr) const noexcept { return mapOffset_; }

  void *map_to_ptr(size_t offset, cl_map_flags flags) noexcept {
    assert(mapPtr_ == nullptr);
    mapOffset_ = offset;
    mapFlags_ = flags;
    if (hostPtr_ && (allocMode_ != alloc_mode::copy_in)) {
      mapPtr_ = static_cast<char *>(hostPtr_) + offset;
    } else {
      // TODO: Allocate only what is needed based on the offset
      mapPtr_ = static_cast<void *>(malloc(this->get_size()));
    }
    return mapPtr_;
  }

  void unmap(void *ptr) noexcept {
    assert(mapPtr_ != nullptr);

    if (mapPtr_ != hostPtr_) {
      free(mapPtr_);
    }
    mapPtr_ = nullptr;
    mapOffset_ = 0;
  }

  cl_map_flags get_map_flags() const noexcept {
    assert(mapPtr_ != nullptr);
    return mapFlags_;
  }
};

/// PI queue mapping on to CUstream objects.
///
struct _pi_queue {
  using native_type = CUstream;

  native_type stream_;
  _pi_context *context_;
  _pi_device *device_;
  pi_queue_properties properties_;
  std::atomic_uint32_t refCount_;

  _pi_queue(CUstream stream, _pi_context *context, _pi_device *device,
            pi_queue_properties properties)
      : stream_{stream}, context_{context}, device_{device},
        properties_{properties}, refCount_{1} {
    cuda_piContextRetain(context_);
    cuda_piDeviceRetain(device_);
  }

  ~_pi_queue() {
    cuda_piContextRelease(context_);
    cuda_piDeviceRelease(device_);
  }

  native_type get() const noexcept { return stream_; };

  _pi_context *get_context() const { return context_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

typedef void (*pfn_notify)(pi_event event, pi_int32 eventCommandStatus,
                           void *userData);
/// PI Event mapping to CUevent
///
class _pi_event {
public:
  using native_type = CUevent;

  pi_result record();

  pi_result wait();

  pi_result start();

  native_type get() const noexcept { return evEnd_; };

  pi_queue get_queue() const noexcept { return queue_; }

  pi_command_type get_command_type() const noexcept { return commandType_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  bool is_recorded() const noexcept { return isRecorded_; }

  bool is_started() const noexcept { return isStarted_; }

  bool is_completed() const noexcept { return isCompleted_; };

  pi_int32 get_execution_status() const noexcept {

    if (!is_recorded()) {
      return PI_EVENT_SUBMITTED;
    }

    if (!is_completed()) {
      return PI_EVENT_RUNNING;
    }
    return PI_EVENT_COMPLETE;
  }

  pi_context get_context() const noexcept { return context_; };

  pi_uint32 increment_reference_count() { return ++refCount_; }

  pi_uint32 decrement_reference_count() { return --refCount_; }

  // Returns the counter time when the associated command(s) were enqueued
  //
  pi_uint64 get_queued_time() const;

  // Returns the counter time when the associated command(s) started execution
  //
  pi_uint64 get_start_time() const;

  // Returns the counter time when the associated command(s) completed
  //
  pi_uint64 get_end_time() const;

  // construct a native CUDA. This maps closely to the underlying CUDA event.
  static pi_event make_native(pi_command_type type, pi_queue queue) {
    return new _pi_event(type, queue->get_context(), queue);
  }

  pi_result release();

  ~_pi_event();

private:
  // This constructor is private to force programmers to use the make_native /
  // make_user static members in order to create a pi_event for CUDA.
  _pi_event(pi_command_type type, pi_context context, pi_queue queue);

  pi_command_type commandType_; // The type of command associated with event.

  std::atomic_uint32_t refCount_; // Event reference count.

  bool isCompleted_; // Signifies whether the operations have completed
                     //

  bool isRecorded_; // Signifies wether a native CUDA event has been recorded
                    // yet.
  bool isStarted_;  // Signifies wether the operation associated with the
                    // PI event has started or not
                    //

  native_type evEnd_; // CUDA event handle. If this _pi_event represents a user
                      // event, this will be nullptr.

  native_type evStart_; // CUDA event handle associated with the start

  native_type evQueued_; // CUDA event handle associated with the time
                         // the command was enqueued

  pi_queue queue_; // pi_queue associated with the event. If this is a user
                   // event, this will be nullptr.

  pi_context context_; // pi_context associated with the event. If this is a
                       // native event, this will be the same context associated
                       // with the queue_ member.
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

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char errorLog_[MAX_LOG_SIZE], infoLog_[MAX_LOG_SIZE];
  std::string buildOptions_;
  pi_program_build_status buildStatus_ = PI_PROGRAM_BUILD_STATUS_NONE;

  _pi_program(pi_context ctxt);
  ~_pi_program();

  pi_result set_binary(const char *binary, size_t binarySizeInBytes);

  pi_result build_program(const char* build_options);

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
  std::string name_;
  pi_context context_;
  pi_program program_;
  std::atomic_uint32_t refCount_;

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

    /// Adds an argument to the kernel.
    /// If the argument existed before, it is replaced.
    /// Otherwise, it is added.
    /// Gaps are filled with empty arguments.
    void add_arg(size_t index, size_t size, const void *arg,
                 size_t localSize = 0) {
      if (index + 1 > indices_.size()) {
        indices_.resize(index + 1);
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
      add_arg(index, sizeof(size_t), (const void *)&(localOffset), size);
    }

    void clear_local_size() {
      std::fill(std::begin(offsetPerIndex_), std::end(offsetPerIndex_), 0);
    }

    args_index_t get_indices() const noexcept { return indices_; }

    pi_uint32 get_local_size() const {
      return std::accumulate(std::begin(offsetPerIndex_),
                             std::end(offsetPerIndex_), 0);
    }
  } args_;

  _pi_kernel(CUfunction func, const char *name, pi_program program,
             pi_context ctxt)
      : function_{func}, name_{name}, context_{ctxt}, program_{program},
        refCount_{1} {
    cuda_piProgramRetain(program_);
    cuda_piContextRetain(context_);
  }

  ~_pi_kernel()
  {
    cuda_piProgramRelease(program_);
    cuda_piContextRelease(context_);
  }

  pi_program get_program() const noexcept { return program_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  native_type get() const noexcept { return function_; };

  pi_context get_context() const noexcept { return context_; };

  const char *get_name() const noexcept { return name_.c_str(); }

  /// Returns the number of arguments.
  /// Note this only returns the current known number of arguments, not the
  /// real one required by the kernel, since this cannot be queried from
  /// the CUDA Driver API
  pi_uint32 get_num_args() const noexcept { return args_.indices_.size(); }

  void set_kernel_arg(int index, size_t size, const void *arg) {
    args_.add_arg(index, size, arg);
  }

  void set_kernel_local_arg(int index, size_t size) {
    args_.add_local_arg(index, size);
  }

  arguments::args_index_t get_arg_indices() const {
    return args_.get_indices();
  }

  pi_uint32 get_local_size() const noexcept { return args_.get_local_size(); }

  void clear_local_size() { args_.clear_local_size(); }
};

// -------------------------------------------------------------
// Helper types and functions
//

#endif // PI_CUDA_HPP

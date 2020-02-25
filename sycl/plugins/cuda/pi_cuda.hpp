//===-- pi_cuda.hpp - CUDA Plugin -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// This source is the definition of the SYCL Plugin Interface
/// (PI). It is the interface between the device-agnostic SYCL runtime layer
/// and underlying "native" runtimes such as OpenCL.

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


}

struct _pi_platform {
};

struct _pi_device {
  using native_type = CUdevice;

  native_type cuDevice_;
  std::atomic_uint32_t refCount_;
  pi_platform platform_;

  _pi_device(native_type cuDevice, pi_platform platform)
      : cuDevice_(cuDevice), refCount_{1}, platform_(platform) {}

  native_type get() const noexcept { return cuDevice_; };

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

struct _pi_context {
  using native_type = CUcontext;

  enum class kind { primary, user_defined } kind_;
  native_type cuContext_;
  _pi_device *deviceId_;
  std::atomic_uint32_t refCount_;

  _pi_context(kind k, CUcontext ctxt, _pi_device *devId)
      : kind_{k}, cuContext_{ctxt}, deviceId_{devId}, refCount_{1} {
    cuda_piDeviceRetain(deviceId_);
  };


  ~_pi_context() { cuda_piDeviceRelease(deviceId_); }

  void invoke_callback()
  {
    std::lock_guard<std::mutex> guard(mutex_);
    for(const auto& callback : destruction_callbacks_)
    {
      callback();
    }
  }

  template<typename Func>
  void register_callback(Func&& callback)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    destruction_callbacks_.emplace_back(std::forward<Func>(callback));
  }

  _pi_device *get_device() const noexcept { return deviceId_; }
  native_type get() const noexcept { return cuContext_; }
  bool is_primary() const noexcept { return kind_ == kind::primary; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
private:
  std::mutex mutex_;
  std::vector<std::function<void(void)>> destruction_callbacks_;
};

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
  enum class alloc_mode { classic, use_host_ptr } allocMode_;

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

  bool is_buffer() const {
    // TODO: Adapt once images are supported.
    return true;
  }
  bool is_sub_buffer() const { return (is_buffer() && (parent_ != nullptr)); }

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
    if (hostPtr_) {
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

  native_type get() const { return stream_; };

  _pi_context *get_context() const { return context_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

class _pi_event {
public:
  using native_type = CUevent;

  pi_result record();

  pi_result wait();

  pi_result start();

  native_type get() const noexcept { return event_; };

  pi_result set_user_event_complete() noexcept {

    if (isCompleted_) {
      return PI_INVALID_OPERATION;
    }

    if (is_user_event()) {
      isRecorded_ = true;
      isCompleted_ = true;
      return PI_SUCCESS;
    }
    return PI_INVALID_EVENT;
  }

  pi_queue get_queue() const noexcept { return queue_; }

  pi_command_type get_command_type() const noexcept { return commandType_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  bool is_recorded() const noexcept { return isRecorded_; }

  bool is_completed() const noexcept { return isCompleted_; }

  bool is_started() const noexcept { return isStarted_; }

  pi_event_status get_execution_status() const noexcept;

  pi_context get_context() const noexcept { return context_; };

  bool is_user_event() const noexcept {
    return get_command_type() == PI_COMMAND_USER;
  }

  bool is_native_event() const noexcept { return !is_user_event(); }

  pi_uint32 increment_reference_count() { return ++refCount_; }

  pi_uint32 decrement_reference_count() { return --refCount_; }

  // Returns the elapsed time in nano-seconds since the command(s)
  // associated with the event have completed
  //
  pi_uint64 get_end_time() const;

  // make a user event. CUDA has no concept of user events, so this
  // functionality is implemented by the CUDA PI implementation.
  static pi_event make_user(pi_context context) {
    return new _pi_event(PI_COMMAND_USER, context, nullptr);
  }

  // construct a native CUDA. This maps closely to the underlying CUDA event.
  static pi_event make_native(pi_command_type type, pi_queue queue) {
    return new _pi_event(type, queue->get_context(), queue);
  }

  ~_pi_event();

private:
  // This constructor is private to force programmers to use the make_native /
  // make_user static members in order to create a pi_event for CUDA.
  _pi_event(pi_command_type type, pi_context context, pi_queue queue);

  pi_command_type commandType_; // The type of command associated with event.

  std::atomic_uint32_t refCount_; // Event reference count.

  std::atomic_bool isCompleted_; // Atomic bool used by user events. Can be
                                 // used to wait for a user event's completion.

  bool isRecorded_; // Signifies wether a native CUDA event has been recorded
                    // yet.
  bool isStarted_; // Signifies wether the operation associated with the
                   // PI event has started or not

  native_type event_; // CUDA event handle. If this _pi_event represents a user
                      // event, this will be nullptr.

  native_type evStart_; // CUDA event handle associated with the start

  pi_queue queue_; // pi_queue associated with the event. If this is a user
                   // event, this will be nullptr.

  pi_context context_; // pi_context associated with the event. If this is a
                       // native event, this will be the same context associated
                       // with the queue_ member.
};

struct _pi_program {
  using native_type = CUmodule;
  native_type module_;
  const char *source_;
  size_t sourceLength_;
  std::atomic_uint32_t refCount_;
  _pi_context *context_;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char errorLog_[MAX_LOG_SIZE], infoLog_[MAX_LOG_SIZE];
  std::string buildOptions_;
  pi_program_build_status buildStatus_ = PI_PROGRAM_BUILD_STATUS_NONE;

  _pi_program(pi_context ctxt);
  ~_pi_program();

  pi_result create_from_source(const char *source, size_t length);

  pi_result build_program(const char* build_options);

  pi_context get_context() const { return context_; };

  native_type get() const { return module_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }
};

struct _pi_kernel {
  using native_type = CUfunction;

  native_type function_;
  std::string name_;
  _pi_context *context_;
  pi_program program_;
  std::atomic_uint32_t refCount_;

  /*
   * Structure that holds the arguments to the kernel.
   * Note earch argument size is known, since it comes
   * from the kernel signature.
   * This is not something you can query in CUDA,
   * so error handling cannot be provided easily.
   */
  struct arguments {
    static constexpr size_t MAX_PARAM_BYTES = 4000u;
    using args_t = std::array<char, MAX_PARAM_BYTES>;
    using args_size_t = std::vector<size_t>;
    using args_index_t = std::vector<void *>;
    args_t storage_;
    args_size_t paramSizes_;
    args_index_t indices_;
    args_size_t offsetPerIndex_;

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

    args_index_t get_indices() const { return indices_; }

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

  native_type get() const { return function_; };

  pi_context get_context() const noexcept { return context_; };


  const char *get_name() const noexcept { return name_.c_str(); }

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

// Checks a CUDA error and returns a PI error code
// May throw
pi_result check_error(CUresult result);

#endif // PI_CUDA_HPP

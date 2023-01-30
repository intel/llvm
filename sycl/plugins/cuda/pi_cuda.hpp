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
struct _pi_platform {
  static CUevent evBase_; // CUDA event used as base counter
  std::vector<std::unique_ptr<_pi_device>> devices_;
};

/// PI device mapping to a CUdevice.
/// Includes an observer pointer to the platform,
/// and implements the reference counting semantics since
/// CUDA objects are not refcounted.
///
struct _pi_device {
private:
  using native_type = CUdevice;

  native_type cuDevice_;
  std::atomic_uint32_t refCount_;
  pi_platform platform_;
  pi_context context_;

  static constexpr pi_uint32 max_work_item_dimensions = 3u;
  size_t max_work_item_sizes[max_work_item_dimensions];
  int max_work_group_size;

public:
  _pi_device(native_type cuDevice, pi_platform platform)
      : cuDevice_(cuDevice), refCount_{1}, platform_(platform) {}

  native_type get() const noexcept { return cuDevice_; };

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  pi_platform get_platform() const noexcept { return platform_; };

  void set_context(pi_context ctx) { context_ = ctx; };

  pi_context get_context() { return context_; };

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

  _pi_context(kind k, CUcontext ctxt, _pi_device *devId,
              bool backend_owns = true)
      : kind_{k}, cuContext_{ctxt}, deviceId_{devId}, refCount_{1},
        has_ownership{backend_owns} {
    deviceId_->set_context(this);
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

  bool backend_has_ownership() const noexcept { return has_ownership; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
  const bool has_ownership;
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
struct _pi_queue {
  using native_type = CUstream;
  static constexpr int default_num_compute_streams = 128;
  static constexpr int default_num_transfer_streams = 64;

  std::vector<native_type> compute_streams_;
  std::vector<native_type> transfer_streams_;
  // delay_compute_ keeps track of which streams have been recently reused and
  // their next use should be delayed. If a stream has been recently reused it
  // will be skipped the next time it would be selected round-robin style. When
  // skipped, its delay flag is cleared.
  std::vector<bool> delay_compute_;
  // keep track of which streams have applied barrier
  std::vector<bool> compute_applied_barrier_;
  std::vector<bool> transfer_applied_barrier_;
  _pi_context *context_;
  _pi_device *device_;
  pi_queue_properties properties_;
  CUevent barrier_event_ = nullptr;
  CUevent barrier_tmp_event_ = nullptr;
  std::atomic_uint32_t refCount_;
  std::atomic_uint32_t eventCount_;
  std::atomic_uint32_t compute_stream_idx_;
  std::atomic_uint32_t transfer_stream_idx_;
  unsigned int num_compute_streams_;
  unsigned int num_transfer_streams_;
  unsigned int last_sync_compute_streams_;
  unsigned int last_sync_transfer_streams_;
  unsigned int flags_;
  // When compute_stream_sync_mutex_ and compute_stream_mutex_ both need to be
  // locked at the same time, compute_stream_sync_mutex_ should be locked first
  // to avoid deadlocks
  std::mutex compute_stream_sync_mutex_;
  std::mutex compute_stream_mutex_;
  std::mutex transfer_stream_mutex_;
  std::mutex barrier_mutex_;
  bool has_ownership_;

  _pi_queue(std::vector<CUstream> &&compute_streams,
            std::vector<CUstream> &&transfer_streams, _pi_context *context,
            _pi_device *device, pi_queue_properties properties,
            unsigned int flags, bool backend_owns = true)
      : compute_streams_{std::move(compute_streams)},
        transfer_streams_{std::move(transfer_streams)},
        delay_compute_(compute_streams_.size(), false),
        compute_applied_barrier_(compute_streams_.size()),
        transfer_applied_barrier_(transfer_streams_.size()), context_{context},
        device_{device}, properties_{properties}, refCount_{1}, eventCount_{0},
        compute_stream_idx_{0}, transfer_stream_idx_{0},
        num_compute_streams_{0}, num_transfer_streams_{0},
        last_sync_compute_streams_{0}, last_sync_transfer_streams_{0},
        flags_(flags), has_ownership_{backend_owns} {
    cuda_piContextRetain(context_);
    cuda_piDeviceRetain(device_);
  }

  ~_pi_queue() {
    cuda_piContextRelease(context_);
    cuda_piDeviceRelease(device_);
  }

  void compute_stream_wait_for_barrier_if_needed(CUstream stream,
                                                 pi_uint32 stream_i);
  void transfer_stream_wait_for_barrier_if_needed(CUstream stream,
                                                  pi_uint32 stream_i);

  // get_next_compute/transfer_stream() functions return streams from
  // appropriate pools in round-robin fashion
  native_type get_next_compute_stream(pi_uint32 *stream_token = nullptr);
  // this overload tries select a stream that was used by one of dependancies.
  // If that is not possible returns a new stream. If a stream is reused it
  // returns a lock that needs to remain locked as long as the stream is in use
  native_type get_next_compute_stream(pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      _pi_stream_guard &guard,
                                      pi_uint32 *stream_token = nullptr);
  native_type get_next_transfer_stream();
  native_type get() { return get_next_compute_stream(); };

  bool has_been_synchronized(pi_uint32 stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<pi_uint32>::max()) {
      return false;
    }
    return last_sync_compute_streams_ >= stream_token;
  }

  bool can_reuse_stream(pi_uint32 stream_token) {
    // stream token not associated with one of the compute streams
    if (stream_token == std::numeric_limits<pi_uint32>::max()) {
      return false;
    }
    // If the command represented by the stream token was not the last command
    // enqueued to the stream we can not reuse the stream - we need to allow for
    // commands enqueued after it and the one we are about to enqueue to run
    // concurrently
    bool is_last_command =
        (compute_stream_idx_ - stream_token) <= compute_streams_.size();
    // If there was a barrier enqueued to the queue after the command
    // represented by the stream token we should not reuse the stream, as we can
    // not take that stream into account for the bookkeeping for the next
    // barrier - such a stream would not be synchronized with. Performance-wise
    // it does not matter that we do not reuse the stream, as the work
    // represented by the stream token is guaranteed to be complete by the
    // barrier before any work we are about to enqueue to the stream will start,
    // so the event does not need to be synchronized with.
    return is_last_command && !has_been_synchronized(stream_token);
  }

  template <typename T> bool all_of(T &&f) {
    {
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(compute_streams_.size()),
                   num_compute_streams_);
      if (!std::all_of(compute_streams_.begin(), compute_streams_.begin() + end,
                       f))
        return false;
    }
    {
      std::lock_guard<std::mutex> transfer_guard(transfer_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(transfer_streams_.size()),
                   num_transfer_streams_);
      if (!std::all_of(transfer_streams_.begin(),
                       transfer_streams_.begin() + end, f))
        return false;
    }
    return true;
  }

  template <typename T> void for_each_stream(T &&f) {
    {
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(compute_streams_.size()),
                   num_compute_streams_);
      for (unsigned int i = 0; i < end; i++) {
        f(compute_streams_[i]);
      }
    }
    {
      std::lock_guard<std::mutex> transfer_guard(transfer_stream_mutex_);
      unsigned int end =
          std::min(static_cast<unsigned int>(transfer_streams_.size()),
                   num_transfer_streams_);
      for (unsigned int i = 0; i < end; i++) {
        f(transfer_streams_[i]);
      }
    }
  }

  template <bool ResetUsed = false, typename T> void sync_streams(T &&f) {
    auto sync_compute = [&f, &streams = compute_streams_,
                         &delay = delay_compute_](unsigned int start,
                                                  unsigned int stop) {
      for (unsigned int i = start; i < stop; i++) {
        f(streams[i]);
        delay[i] = false;
      }
    };
    auto sync_transfer = [&f, &streams = transfer_streams_](unsigned int start,
                                                            unsigned int stop) {
      for (unsigned int i = start; i < stop; i++) {
        f(streams[i]);
      }
    };
    {
      unsigned int size = static_cast<unsigned int>(compute_streams_.size());
      std::lock_guard compute_sync_guard(compute_stream_sync_mutex_);
      std::lock_guard<std::mutex> compute_guard(compute_stream_mutex_);
      unsigned int start = last_sync_compute_streams_;
      unsigned int end = num_compute_streams_ < size
                             ? num_compute_streams_
                             : compute_stream_idx_.load();
      if (ResetUsed) {
        last_sync_compute_streams_ = end;
      }
      if (end - start >= size) {
        sync_compute(0, size);
      } else {
        start %= size;
        end %= size;
        if (start <= end) {
          sync_compute(start, end);
        } else {
          sync_compute(start, size);
          sync_compute(0, end);
        }
      }
    }
    {
      unsigned int size = static_cast<unsigned int>(transfer_streams_.size());
      if (size > 0) {
        std::lock_guard<std::mutex> transfer_guard(transfer_stream_mutex_);
        unsigned int start = last_sync_transfer_streams_;
        unsigned int end = num_transfer_streams_ < size
                               ? num_transfer_streams_
                               : transfer_stream_idx_.load();
        if (ResetUsed) {
          last_sync_transfer_streams_ = end;
        }
        if (end - start >= size) {
          sync_transfer(0, size);
        } else {
          start %= size;
          end %= size;
          if (start <= end) {
            sync_transfer(start, end);
          } else {
            sync_transfer(start, size);
            sync_transfer(0, end);
          }
        }
      }
    }
  }

  _pi_context *get_context() const { return context_; };

  _pi_device *get_device() const { return device_; };

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  pi_uint32 get_next_event_id() noexcept { return ++eventCount_; }

  bool backend_has_ownership() const noexcept { return has_ownership_; }
};

typedef void (*pfn_notify)(pi_event event, pi_int32 eventCommandStatus,
                           void *userData);
/// PI Event mapping to CUevent
///
struct _pi_event {
public:
  using native_type = CUevent;

  pi_result record();

  pi_result wait();

  pi_result start();

  native_type get() const noexcept { return evEnd_; };

  pi_queue get_queue() const noexcept { return queue_; }

  CUstream get_stream() const noexcept { return stream_; }

  pi_uint32 get_compute_stream_token() const noexcept { return streamToken_; }

  pi_command_type get_command_type() const noexcept { return commandType_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

  bool is_recorded() const noexcept { return isRecorded_; }

  bool is_started() const noexcept { return isStarted_; }

  bool is_completed() const noexcept;

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

  pi_uint32 get_event_id() const noexcept { return eventId_; }

  bool backend_has_ownership() const noexcept { return has_ownership_; }

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
  static pi_event
  make_native(pi_command_type type, pi_queue queue, CUstream stream,
              pi_uint32 stream_token = std::numeric_limits<pi_uint32>::max()) {
    return new _pi_event(type, queue->get_context(), queue, stream,
                         stream_token);
  }

  static pi_event make_with_native(pi_context context, CUevent eventNative) {
    return new _pi_event(context, eventNative);
  }

  pi_result release();

  ~_pi_event();

private:
  // This constructor is private to force programmers to use the make_native /
  // make_user static members in order to create a pi_event for CUDA.
  _pi_event(pi_command_type type, pi_context context, pi_queue queue,
            CUstream stream, pi_uint32 stream_token);

  // This constructor is private to force programmers to use the
  // make_with_native for event introp
  _pi_event(pi_context context, CUevent eventNative);

  pi_command_type commandType_; // The type of command associated with event.

  std::atomic_uint32_t refCount_; // Event reference count.

  bool has_ownership_; // Signifies if event owns the native type.

  bool hasBeenWaitedOn_; // Signifies whether the event has been waited
                         // on through a call to wait(), which implies
                         // that it has completed.

  bool isRecorded_; // Signifies wether a native CUDA event has been recorded
                    // yet.
  bool isStarted_;  // Signifies wether the operation associated with the
                    // PI event has started or not
                    //

  pi_uint32 streamToken_;
  pi_uint32 eventId_; // Queue identifier of the event.

  native_type evEnd_; // CUDA event handle. If this _pi_event represents a user
                      // event, this will be nullptr.

  native_type evStart_; // CUDA event handle associated with the start

  native_type evQueued_; // CUDA event handle associated with the time
                         // the command was enqueued

  pi_queue queue_; // pi_queue associated with the event. If this is a user
                   // event, this will be nullptr.

  CUstream stream_; // CUstream associated with the event. If this is a user
                    // event, this will be uninitialized.

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
        this, ctxt->get_device(), PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
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

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

#include <ur/adapters/cuda/context.hpp>
#include <ur/adapters/cuda/device.hpp>
#include <ur/adapters/cuda/kernel.hpp>
#include <ur/adapters/cuda/platform.hpp>
#include <ur/adapters/cuda/program.hpp>
#include <ur/adapters/cuda/event.hpp>
#include <ur/adapters/cuda/queue.hpp>
#include <ur/adapters/cuda/sampler.hpp>
#include <ur/adapters/cuda/memory.hpp>

// Share code between the PI Plugin and UR Adapter
#include <pi2ur.hpp>

extern "C" {

/// \cond IGNORE_BLOCK_IN_DOXYGEN
pi_result cuda_piMemRetain(pi_mem);
pi_result cuda_piMemRelease(pi_mem);
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
struct _pi_mem  : ur_mem_handle_t_ {
  using ur_mem_handle_t_::ur_mem_handle_t_;
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
struct _pi_program : ur_program_handle_t_ {
  using ur_program_handle_t_::ur_program_handle_t_;
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
struct _pi_kernel : ur_kernel_handle_t_ {
  using ur_kernel_handle_t_::ur_kernel_handle_t_;
};

/// Implementation of samplers for CUDA
///
/// Sampler property layout:
/// | 31 30 ... 6 5 |      4 3 2      |     1      |         0        |
/// |      N/A      | addressing mode | fiter mode | normalize coords |
struct _pi_sampler : ur_sampler_handle_t_ {
  using ur_sampler_handle_t_::ur_sampler_handle_t_;
};

// -------------------------------------------------------------
// Helper types and functions
//

#endif // PI_CUDA_HPP

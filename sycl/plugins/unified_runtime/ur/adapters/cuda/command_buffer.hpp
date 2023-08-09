//===--------- command_buffer.hpp - CUDA Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur/ur.hpp>
#include <ur_api.h>

#include <cuda.h>

struct ur_exp_command_buffer_handle_t_ {

  ur_exp_command_buffer_handle_t_(ur_context_handle_t hContext,
                                  ur_device_handle_t hDevice);

  ~ur_exp_command_buffer_handle_t_();

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command buffer
  ur_device_handle_t Device;
  // Cuda Graph handle
  CUgraph cudaGraph;
  // Cuda Graph Exec handle
  CUgraphExec cudaGraphExec;
  // Atomic variable counting the number of reference to this command_buffer
  // using std::atomic prevents data race when incrementing/decrementing.
  std::atomic_uint32_t RefCount;

  // Used when retaining an object.
  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }
  // Used when releasing an object.
  uint32_t decrementReferenceCount() noexcept { return --RefCount; }
  uint32_t getReferenceCount() const noexcept { return RefCount; }
  // This method allows to guard a code which needs to be executed when object's
  // ref count becomes zero after release. It is important to notice that only a
  // single thread can pass through this check. This is true because of several
  // reasons:
  //   1. Decrement operation is executed atomically.
  //   2. It is not allowed to retain an object after its refcount reaches zero.
  //   3. It is not allowed to release an object more times than the value of
  //   the ref count.
  // 2. and 3. basically means that we can't use an object at all as soon as its
  // refcount reaches zero. Using this check guarantees that code for deleting
  // an object and releasing its resources is executed once by a single thread
  // and we don't need to use any mutexes to guard access to this object in the
  // scope after this check. Of course if we access another objects in this code
  // (not the one which is being deleted) then access to these objects must be
  // guarded, for example with a mutex.
  bool decrementAndTestReferenceCount() { return --RefCount == 0; }
};

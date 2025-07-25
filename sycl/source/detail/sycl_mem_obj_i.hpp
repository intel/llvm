//==------------ sycl_mem_obj_i.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include <memory>

namespace sycl {
inline namespace _V1 {

namespace detail {

class event_impl;
class context_impl;
struct MemObjRecord;

using EventImplPtr = std::shared_ptr<detail::event_impl>;

// The class serves as an interface in the scheduler for all SYCL memory
// objects.
class SYCLMemObjI {
public:
  virtual ~SYCLMemObjI() = default;

  enum MemObjType { Buffer = 0, Image = 1, Undefined = 2 };

  virtual MemObjType getType() const = 0;

  // The method allocates memory for the SYCL memory object. The size of
  // allocation will be taken from the size of SYCL memory object.
  // If the memory returned cannot be used right away InteropEvent will
  // point to event that should be waited before using the memory.
  // InitFromUserData indicates that the returned memory should be intialized
  // with the data provided by user(if any). Usually it should happen on the
  // first allocation of memory for the memory object.
  // Non null HostPtr requires allocation to be made with USE_HOST_PTR property.
  // Method returns a pointer to host allocation if Context is host one and
  // cl_mem obect if not.
  virtual void *allocateMem(context_impl *Context, bool InitFromUserData,
                            void *HostPtr, ur_event_handle_t &InteropEvent) = 0;

  // Should be used for memory object created without use_host_ptr property.
  virtual void *allocateHostMem() = 0;

  // Ptr must be a pointer returned by allocateMem for the same context.
  // If Context is a device context and Ptr is a host pointer exception will be
  // thrown. And it's undefined behaviour if Context is a host context and Ptr
  // is a device pointer.
  virtual void releaseMem(context_impl *Context, void *Ptr) = 0;

  // Ptr must be a pointer returned by allocateHostMem.
  virtual void releaseHostMem(void *Ptr) = 0;

  // Returns size of object in bytes
  virtual size_t getSizeInBytes() const noexcept = 0;

  virtual bool isInterop() const = 0;

  virtual bool hasUserDataPtr() const = 0;

  virtual bool isHostPointerReadOnly() const = 0;

  virtual bool usesPinnedHostMemory() const = 0;

  // Returns the context which is passed if a memory object is created using
  // interoperability constructor, nullptr otherwise.
  virtual detail::context_impl *getInteropContext() const = 0;

protected:
  // Pointer to the record that contains the memory commands. This is managed
  // by the scheduler.
  // fixme replace with std::unique_ptr once it is implemented. Standard
  // unique_ptr requires knowlege of sizeof(MemObjRecord) at compile time
  // which is unavailable.
  std::shared_ptr<MemObjRecord> MRecord;
  friend class Scheduler;
  friend class ExecCGCommand;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

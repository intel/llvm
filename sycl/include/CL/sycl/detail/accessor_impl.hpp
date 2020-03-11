//==------------ accessor_impl.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/sycl_mem_obj_i.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class Command;

// The class describes a requirement to access a SYCL memory object such as
// sycl::buffer and sycl::image. For example, each accessor used in a kernel,
// except one with access target "local", adds such requirement for the command
// group.

template <int Dims> class AccessorImplDevice {
public:
  AccessorImplDevice() = default;
  AccessorImplDevice(id<Dims> Offset, range<Dims> AccessRange,
                     range<Dims> MemoryRange)
      : Offset(Offset), AccessRange(AccessRange), MemRange(MemoryRange) {}

  id<Dims> Offset;
  range<Dims> AccessRange;
  range<Dims> MemRange;

  bool operator==(const AccessorImplDevice &Rhs) const {
    return (Offset == Rhs.Offset && AccessRange == Rhs.AccessRange &&
            MemRange == Rhs.MemRange);
  }
};

template <int Dims> class LocalAccessorBaseDevice {
public:
  LocalAccessorBaseDevice(sycl::range<Dims> Size)
      : AccessRange(Size),
        MemRange(InitializedVal<Dims, range>::template get<0>()) {}
  // TODO: Actually we need only one field here, but currently compiler requires
  // all of them.
  range<Dims> AccessRange;
  range<Dims> MemRange;
  id<Dims> Offset;

  bool operator==(const LocalAccessorBaseDevice &Rhs) const {
    return (AccessRange == Rhs.AccessRange);
  }
};

// Forward declaration
class AccessorImplHost;

using AccessorImplPtr = shared_ptr_class<AccessorImplHost>;

class AccessorBaseHost {
public:
  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, detail::SYCLMemObjI *SYCLMemObject,
                   int Dims, int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false);

protected:
  id<3> &getOffset() {
    if (!MCacheInitialized)
      initializeCache();
    return MCachedOffset;
  }
  range<3> &getAccessRange() {
    if (!MCacheInitialized)
      initializeCache();
    return MCachedAccessRange;
  }
  range<3> &getMemoryRange() {
    if (!MCacheInitialized)
      initializeCache();
    return MCachedMemoryRange;
  }
  void *getPtr() {
    if (!MCacheInitialized)
      initializeCache();
    return MCachedPtr;
  }

  unsigned int getElemSize() const {
    if (MCacheInitialized)
      return MCachedElemSize;
    return getConstElemSize();
  }

  const id<3> &getOffset() const {
    if (MCacheInitialized)
      return MCachedOffset;
    return getConstOffset();
  }
  const range<3> &getAccessRange() const {
    if (MCacheInitialized)
      return MCachedAccessRange;
    return getConstAccessRange();
  }
  const range<3> &getMemoryRange() const {
    if (MCacheInitialized)
      return MCachedMemoryRange;
    return getConstMemoryRange();
  }
  void *getPtr() const {
    if (MCacheInitialized)
      return MCachedPtr;
    return getConstPtr();
  }

  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  AccessorImplPtr impl;

private:
  void initializeCache();

  unsigned int getConstElemSize() const;

  const id<3> &getConstOffset() const;
  const range<3> &getConstAccessRange() const;
  const range<3> &getConstMemoryRange() const;
  void *getConstPtr() const;

  bool MCacheInitialized = false;
  id<3> MCachedOffset{};
  range<3> MCachedAccessRange{0, 0, 0};
  range<3> MCachedMemoryRange{0, 0, 0};
  void *MCachedPtr{};
  unsigned int MCachedElemSize{};
};

class LocalAccessorImplHost {
public:
  LocalAccessorImplHost(sycl::range<3> Size, int Dims, int ElemSize);

  sycl::range<3> MSize;
  int MDims;
  int MElemSize;
  std::vector<char> MMem;

  bool PerWI = false;
  size_t LocalMemSize;
  size_t MaxWGSize;
  void resize(size_t LocalSize, size_t GlobalSize);
};

using LocalAccessorImplPtr = shared_ptr_class<LocalAccessorImplHost>;

class LocalAccessorBaseHost {
public:
  LocalAccessorBaseHost(sycl::range<3> Size, int Dims, int ElemSize);

  sycl::range<3> &getSize() { return impl->MSize; }
  const sycl::range<3> &getSize() const { return impl->MSize; }
  void *getPtr() { return impl->MMem.data(); }
  void *getPtr() const {
    return const_cast<void *>(reinterpret_cast<void *>(impl->MMem.data()));
  }

  int getNumOfDims() { return impl->MDims; }
  int getElementSize() { return impl->MElemSize; }

protected:
  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  shared_ptr_class<LocalAccessorImplHost> impl;
};

using Requirement = AccessorImplHost;

void addHostAccessorAndWait(Requirement *Req);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

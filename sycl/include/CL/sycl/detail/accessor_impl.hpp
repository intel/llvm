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

#if __has_attribute(const)
#define CONST_ATTR __attribute__((const))
#else
#define CONST_ATTR
#endif

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
  CONST_ATTR unsigned int getElemSize() const;

  CONST_ATTR const id<3> &getOffset() const;
  CONST_ATTR const range<3> &getAccessRange() const;
  CONST_ATTR const range<3> &getMemoryRange() const;

  CONST_ATTR void *getPtr() const;

  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  AccessorImplPtr impl;

private:
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

#undef CONST_ATTR
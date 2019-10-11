//==------------ accessor_impl.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/sycl_mem_obj_i.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>

#include <memory>

namespace cl {
namespace sycl {
namespace detail {

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
    return (Offset == Rhs.Offset &&
            AccessRange == Rhs.AccessRange &&
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

class AccessorImplHost {
public:
  AccessorImplHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, detail::SYCLMemObjI *SYCLMemObject,
                   int Dims, int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false)
      : MOffset(Offset), MAccessRange(AccessRange), MMemoryRange(MemoryRange),
        MAccessMode(AccessMode), MSYCLMemObj(SYCLMemObject), MDims(Dims),
        MElemSize(ElemSize), MOffsetInBytes(OffsetInBytes),
        MIsSubBuffer(IsSubBuffer) {}

  ~AccessorImplHost() {
    if (BlockingEvent)
      BlockingEvent->setComplete();
  }
  AccessorImplHost(const AccessorImplHost &Other)
      : MOffset(Other.MOffset), MAccessRange(Other.MAccessRange),
        MMemoryRange(Other.MMemoryRange), MAccessMode(Other.MAccessMode),
        MSYCLMemObj(Other.MSYCLMemObj), MDims(Other.MDims),
        MElemSize(Other.MElemSize), MOffsetInBytes(Other.MOffsetInBytes),
        MIsSubBuffer(Other.MIsSubBuffer) {}

  id<3> MOffset;
  // The size of accessing region.
  range<3> MAccessRange;
  // The size of memory object this requirement is created for.
  range<3> MMemoryRange;
  access::mode MAccessMode;

  detail::SYCLMemObjI *MSYCLMemObj;

  unsigned int MDims;
  unsigned int MElemSize;
  unsigned int MOffsetInBytes;
  bool MIsSubBuffer;

  void *MData = nullptr;

  EventImplPtr BlockingEvent;
};

using AccessorImplPtr = std::shared_ptr<AccessorImplHost>;

class AccessorBaseHost {
public:
  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, detail::SYCLMemObjI *SYCLMemObject,
                   int Dims, int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false) {
    impl = std::make_shared<AccessorImplHost>(
        Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject, Dims,
        ElemSize, OffsetInBytes, IsSubBuffer);
  }

protected:
  id<3> &getOffset() { return impl->MOffset; }
  range<3> &getAccessRange() { return impl->MAccessRange; }
  range<3> &getMemoryRange() { return impl->MMemoryRange; }
  void *getPtr() { return impl->MData; }
  unsigned int getElemSize() const { return impl->MElemSize; }

  const id<3> &getOffset() const { return impl->MOffset; }
  const range<3> &getAccessRange() const { return impl->MAccessRange; }
  const range<3> &getMemoryRange() const { return impl->MMemoryRange; }
  void *getPtr() const { return const_cast<void *>(impl->MData); }

  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  AccessorImplPtr impl;
};

class LocalAccessorImplHost {
public:
  LocalAccessorImplHost(sycl::range<3> Size, int Dims, int ElemSize)
      : MSize(Size), MDims(Dims), MElemSize(ElemSize),
        MMem(Size[0] * Size[1] * Size[2] * ElemSize) {}

  sycl::range<3> MSize;
  int MDims;
  int MElemSize;
  std::vector<char> MMem;

  bool PerWI = false;
  size_t LocalMemSize;
  size_t MaxWGSize;
  void resize(size_t LocalSize, size_t GlobalSize) {
    if (GlobalSize != 1 && LocalSize != 1) {
      // If local size is not specified then work group size is chosen by
      // runtime. That is why try to allocate based on max work group size or
      // global size. In the worst case allocate 80% of local memory.
      size_t MinEstWGSize = LocalSize ? LocalSize : GlobalSize;
      MinEstWGSize = MinEstWGSize > MaxWGSize ? MaxWGSize : MinEstWGSize;
      size_t NewSize = MinEstWGSize * MSize[0];
      MSize[0] =
          NewSize > 8 * LocalMemSize / 10 ? 8 * LocalMemSize / 10 : NewSize;
      MMem.resize(NewSize * MElemSize);
    }
  }
};

class LocalAccessorBaseHost {
public:
  LocalAccessorBaseHost(sycl::range<3> Size, int Dims, int ElemSize) {
    impl = std::make_shared<LocalAccessorImplHost>(Size, Dims, ElemSize);
  }
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

  std::shared_ptr<LocalAccessorImplHost> impl;
};

using Requirement = AccessorImplHost;

} // namespace detail
} // namespace sycl
} // namespace cl

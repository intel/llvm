//==------------ accessor_impl.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/sycl_mem_obj_i.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {
// Forward declare a "back-door" access class to support ESIMD.
class AccessorPrivateProxy;
} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext

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

class __SYCL_EXPORT AccessorImplHost {
public:
  AccessorImplHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, detail::SYCLMemObjI *SYCLMemObject,
                   int Dims, int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false, bool IsESIMDAcc = false)
      : MOffset(Offset), MAccessRange(AccessRange), MMemoryRange(MemoryRange),
        MAccessMode(AccessMode), MSYCLMemObj(SYCLMemObject), MDims(Dims),
        MElemSize(ElemSize), MOffsetInBytes(OffsetInBytes),
        MIsSubBuffer(IsSubBuffer), MIsESIMDAcc(IsESIMDAcc) {}

  ~AccessorImplHost();

  AccessorImplHost(const AccessorImplHost &Other)
      : MOffset(Other.MOffset), MAccessRange(Other.MAccessRange),
        MMemoryRange(Other.MMemoryRange), MAccessMode(Other.MAccessMode),
        MSYCLMemObj(Other.MSYCLMemObj), MDims(Other.MDims),
        MElemSize(Other.MElemSize), MOffsetInBytes(Other.MOffsetInBytes),
        MIsSubBuffer(Other.MIsSubBuffer), MIsESIMDAcc(Other.MIsESIMDAcc) {}

  // The resize method provides a way to change the size of the
  // allocated memory and corresponding properties for the accessor.
  // These are normally fixed for the accessor, but this capability
  // is needed to support the stream class.
  // Stream implementation creates an accessor with initial size for
  // work item. But the number of work items is not available during
  // stream construction. The resize method allows to update the accessor
  // as the information becomes available to the handler.

  void resize(size_t GlobalSize);

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

  Command *MBlockedCmd = nullptr;

  bool PerWI = false;

  // Outdated, leaving to preserve ABI.
  // TODO: Remove during next major release.
  bool MIsESIMDAcc;
};

using AccessorImplPtr = std::shared_ptr<AccessorImplHost>;

class AccessorBaseHost {
public:
  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, detail::SYCLMemObjI *SYCLMemObject,
                   int Dims, int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false) {
    impl = std::shared_ptr<AccessorImplHost>(new AccessorImplHost(
        Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject, Dims,
        ElemSize, OffsetInBytes, IsSubBuffer));
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

  template <typename, int, access::mode, access::target, access::placeholder,
            typename>
  friend class accessor;

  AccessorImplPtr impl;

private:
  friend class sycl::ext::intel::experimental::esimd::detail::
      AccessorPrivateProxy;
};

class __SYCL_EXPORT LocalAccessorImplHost {
public:
  // Allocate ElemSize more data to have sufficient padding to enforce
  // alignment.
  LocalAccessorImplHost(sycl::range<3> Size, int Dims, int ElemSize)
      : MSize(Size), MDims(Dims), MElemSize(ElemSize),
        MMem(Size[0] * Size[1] * Size[2] * ElemSize + ElemSize) {}

  sycl::range<3> MSize;
  int MDims;
  int MElemSize;
  std::vector<char> MMem;
};

using LocalAccessorImplPtr = std::shared_ptr<LocalAccessorImplHost>;

class LocalAccessorBaseHost {
public:
  LocalAccessorBaseHost(sycl::range<3> Size, int Dims, int ElemSize) {
    impl = std::shared_ptr<LocalAccessorImplHost>(
        new LocalAccessorImplHost(Size, Dims, ElemSize));
  }
  sycl::range<3> &getSize() { return impl->MSize; }
  const sycl::range<3> &getSize() const { return impl->MSize; }
  void *getPtr() {
    // Const cast this in order to call the const getPtr.
    return const_cast<const LocalAccessorBaseHost *>(this)->getPtr();
  }
  void *getPtr() const {
    char *ptr = impl->MMem.data();

    // Align the pointer to MElemSize.
    size_t val = reinterpret_cast<size_t>(ptr);
    if (val % impl->MElemSize != 0) {
      ptr += impl->MElemSize - val % impl->MElemSize;
    }

    return ptr;
  }

  int getNumOfDims() { return impl->MDims; }
  int getElementSize() { return impl->MElemSize; }

protected:
  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  std::shared_ptr<LocalAccessorImplHost> impl;
};

using Requirement = AccessorImplHost;

void __SYCL_EXPORT addHostAccessorAndWait(Requirement *Req);

#if __cplusplus >= 201703L

template <typename MayBeTag1, typename MayBeTag2>
constexpr access::mode deduceAccessMode() {
  // property_list = {} is not properly detected by deduction guide,
  // when parameter is passed without curly braces: access(buffer, no_init)
  // thus simplest approach is to check 2 last arguments for being a tag
  if constexpr (std::is_same<MayBeTag1,
                             mode_tag_t<access::mode::read>>::value ||
                std::is_same<MayBeTag2,
                             mode_tag_t<access::mode::read>>::value) {
    return access::mode::read;
  }

  if constexpr (std::is_same<MayBeTag1,
                             mode_tag_t<access::mode::write>>::value ||
                std::is_same<MayBeTag2,
                             mode_tag_t<access::mode::write>>::value) {
    return access::mode::write;
  }

  if constexpr (
      std::is_same<MayBeTag1,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value ||
      std::is_same<MayBeTag2,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value) {
    return access::mode::read;
  }

  return access::mode::read_write;
}

template <typename MayBeTag1, typename MayBeTag2>
constexpr access::target deduceAccessTarget(access::target defaultTarget) {
  if constexpr (
      std::is_same<MayBeTag1,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value ||
      std::is_same<MayBeTag2,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value) {
    return access::target::constant_buffer;
  }

  return defaultTarget;
}

#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

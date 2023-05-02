//==------------ accessor_impl.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/accessor.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/id.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>
#include <sycl/stl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
template <typename, int, access::mode, access::target, access::placeholder,
          typename>
class accessor;

namespace ext {
namespace intel {
namespace esimd {
namespace detail {
// Forward declare a "back-door" access class to support ESIMD.
class AccessorPrivateProxy;
} // namespace detail
} // namespace esimd
} // namespace intel
} // namespace ext

namespace detail {

class SYCLMemObjI;

class Command;

class __SYCL_EXPORT AccessorImplHost {
public:
  // TODO: Remove when ABI break is allowed.
  AccessorImplHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {})
      : MAccData(Offset, AccessRange, MemoryRange), MAccessMode(AccessMode),
        MSYCLMemObj((detail::SYCLMemObjI *)SYCLMemObject), MDims(Dims),
        MElemSize(ElemSize), MOffsetInBytes(OffsetInBytes),
        MIsSubBuffer(IsSubBuffer), MPropertyList(PropertyList),
        MIsPlaceH(false) {}

  AccessorImplHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, bool IsPlaceH, int OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {})
      : MAccData(Offset, AccessRange, MemoryRange), MAccessMode(AccessMode),
        MSYCLMemObj((detail::SYCLMemObjI *)SYCLMemObject), MDims(Dims),
        MElemSize(ElemSize), MOffsetInBytes(OffsetInBytes),
        MIsSubBuffer(IsSubBuffer), MPropertyList(PropertyList),
        MIsPlaceH(IsPlaceH) {}

  ~AccessorImplHost();

  AccessorImplHost(const AccessorImplHost &Other)
      : MAccData(Other.MAccData), MAccessMode(Other.MAccessMode),
        MSYCLMemObj(Other.MSYCLMemObj), MDims(Other.MDims),
        MElemSize(Other.MElemSize), MOffsetInBytes(Other.MOffsetInBytes),
        MIsSubBuffer(Other.MIsSubBuffer), MPropertyList(Other.MPropertyList),
        MIsPlaceH(Other.MIsPlaceH) {}

  AccessorImplHost &operator=(const AccessorImplHost &Other) {
    MAccData = Other.MAccData;
    MAccessMode = Other.MAccessMode;
    MSYCLMemObj = Other.MSYCLMemObj;
    MDims = Other.MDims;
    MElemSize = Other.MElemSize;
    MOffsetInBytes = Other.MOffsetInBytes;
    MIsSubBuffer = Other.MIsSubBuffer;
    MPropertyList = Other.MPropertyList;
    MIsPlaceH = Other.MIsPlaceH;
    return *this;
  }

  // The resize method provides a way to change the size of the
  // allocated memory and corresponding properties for the accessor.
  // These are normally fixed for the accessor, but this capability
  // is needed to support the stream class.
  // Stream implementation creates an accessor with initial size for
  // work item. But the number of work items is not available during
  // stream construction. The resize method allows to update the accessor
  // as the information becomes available to the handler.

  void resize(size_t GlobalSize);

  detail::AccHostDataT MAccData;

  id<3> &MOffset = MAccData.MOffset;
  // The size of accessing region.
  range<3> &MAccessRange = MAccData.MAccessRange;
  // The size of memory object this requirement is created for.
  range<3> &MMemoryRange = MAccData.MMemoryRange;
  access::mode MAccessMode;

  detail::SYCLMemObjI *MSYCLMemObj;

  unsigned int MDims;
  unsigned int MElemSize;
  unsigned int MOffsetInBytes;
  bool MIsSubBuffer;

  void *&MData = MAccData.MData;

  Command *MBlockedCmd = nullptr;

  bool PerWI = false;

  // To preserve runtime properties
  property_list MPropertyList;

  // Placeholder flag
  bool MIsPlaceH;
};

using AccessorImplPtr = std::shared_ptr<AccessorImplHost>;

class __SYCL_EXPORT LocalAccessorImplHost {
public:
  // Allocate ElemSize more data to have sufficient padding to enforce
  // alignment.
  LocalAccessorImplHost(sycl::range<3> Size, int Dims, int ElemSize,
                        const property_list &PropertyList)
      : MSize(Size), MDims(Dims), MElemSize(ElemSize),
        MMem(Size[0] * Size[1] * Size[2] * ElemSize + ElemSize),
        MPropertyList(PropertyList) {}

  sycl::range<3> MSize;
  int MDims;
  int MElemSize;
  std::vector<char> MMem;
  property_list MPropertyList;
};

using LocalAccessorImplPtr = std::shared_ptr<LocalAccessorImplHost>;

using Requirement = AccessorImplHost;

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

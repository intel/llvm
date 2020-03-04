//==------------- accessor_impl_host.hpp - SYCL standard source file -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// Forward declaration
class Command;
class SYCLMemObjI;

class AccessorImplHost {
public:
  AccessorImplHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, detail::SYCLMemObjI *SYCLMemObject,
                   int Dims, int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false);

  ~AccessorImplHost();

  AccessorImplHost(const AccessorImplHost &Other);

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
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

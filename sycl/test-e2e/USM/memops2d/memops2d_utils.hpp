//==------------------- memops2d_utils.hpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

enum OperationPath { Expanded, ExpandedDependsOn, ShortcutEventList };

enum Alloc {
  Device = (int)usm::alloc::device,
  Host = (int)usm::alloc::host,
  Shared = (int)usm::alloc::shared,
  DirectHost
};

std::string operationPathToString(OperationPath PathKind) {
  switch (PathKind) {
  case Expanded:
    return "no shortcut and no depends_on";
  case ExpandedDependsOn:
    return "no shortcut";
  case ShortcutEventList:
    return "shortcut with dependency event list";
  default:
    return "UNKNOWN";
  }
}

std::string allocTypeToString(Alloc AllocKind) {
  switch (AllocKind) {
  case Alloc::Device:
    return "device USM allocation";
  case Alloc::Host:
    return "host USM allocation";
  case Alloc::Shared:
    return "shared USM allocation";
  case Alloc::DirectHost:
    return "direct host allocation";
  default:
    return "UNKNOWN";
  }
}

template <typename T, Alloc AllocKind> T *allocate(size_t N, sycl::queue &Q) {
  switch (AllocKind) {
  case Alloc::Device:
  case Alloc::Host:
  case Alloc::Shared:
    return sycl::malloc<T>(N, Q, (usm::alloc)AllocKind);
  case Alloc::DirectHost:
    return (T *)malloc(N * sizeof(T));
  default:
    return nullptr;
  }
}

template <Alloc AllocKind> void free(void *Ptr, sycl::queue &Q) {
  switch (AllocKind) {
  case Alloc::Device:
  case Alloc::Host:
  case Alloc::Shared:
    sycl::free(Ptr, Q);
    return;
  case Alloc::DirectHost:
    free(Ptr);
    return;
  default:
    return;
  }
}

template <Alloc AllocKind, typename T>
sycl::event fill(sycl::queue &Q, void *ptr, const T &pattern, size_t count) {
  switch (AllocKind) {
  case Alloc::Device:
  case Alloc::Shared:
    return Q.fill(ptr, pattern, count);
  case Alloc::Host:
  case Alloc::DirectHost:
    std::fill(static_cast<T *>(ptr), static_cast<T *>(ptr) + count, pattern);
    return sycl::event();
  default:
    return sycl::event();
  }
}

template <Alloc AllocKind>
sycl::event memset(sycl::queue &Q, void *ptr, int value, size_t numBytes) {
  return fill<AllocKind, uint8_t>(Q, ptr, (uint8_t)value, numBytes);
}

template <Alloc AllocKind, typename T, typename Functor>
sycl::event fill_with(sycl::queue &Q, T *ptr, size_t count, Functor func) {
  switch (AllocKind) {
  case Alloc::Device:
  case Alloc::Shared:
    return Q.parallel_for(count, [=](item<1> Id) { ptr[Id] = func(Id[0]); });
  case Alloc::Host:
  case Alloc::DirectHost:
    for (size_t I = 0; I < count; ++I)
      ptr[I] = func(I);
    return sycl::event();
  default:
    return sycl::event();
  }
}

template <Alloc SrcAllocKind, typename T>
sycl::event copy_to_host(sycl::queue &Q, T *src_ptr, T *host_dst_ptr,
                         size_t count) {
  switch (SrcAllocKind) {
  case Alloc::Device:
  case Alloc::Shared:
    return Q.copy(src_ptr, host_dst_ptr, count);
  case Alloc::Host:
  case Alloc::DirectHost:
    std::copy(src_ptr, src_ptr + count, host_dst_ptr);
    return sycl::event();
  default:
    return sycl::event();
  }
}

struct TestStruct {
  int a;
  char b;

  bool operator==(const TestStruct &RHS) const {
    return a == RHS.a && b == RHS.b;
  }
  bool operator!=(const TestStruct &RHS) const { return !(*this == RHS); }
};

std::ostream &operator<<(std::ostream &Out, const TestStruct &RHS) {
  Out << '{' << RHS.a << ',' << RHS.b << '}';
  return Out;
}

template <Alloc SrcAllocKind, Alloc DstAllocKind, OperationPath PathKind,
          typename T>
bool checkResult(T &Result, T &Expected, size_t Index,
                 std::string_view TestName) {
  if (Result != Expected) {
    std::cout << TestName << " (" << allocTypeToString(SrcAllocKind);
    if constexpr (SrcAllocKind != DstAllocKind)
      std::cout << " to " << allocTypeToString(DstAllocKind);
    std::cout << ", " << operationPathToString(PathKind) << ")\nValue at "
              << Index << " did not match the expected value; " << Result
              << " != " << Expected << std::endl;
    return false;
  }
  return true;
}

template <Alloc AllocKind, OperationPath PathKind, typename T>
bool checkResult(T &Result, T &Expected, size_t Index,
                 std::string_view TestName) {
  return checkResult<AllocKind, AllocKind, PathKind, T>(Result, Expected, Index,
                                                        TestName);
}

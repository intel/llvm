//==--- kernel_program_cache.cpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/kernel_program_cache.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {
KernelProgramCache::~KernelProgramCache() {
  for (auto &ProgIt : MCachedPrograms) {
    ProgramWithBuildStateT &ProgWithState = ProgIt.second;
    PiProgramT *ToBeDeleted = ProgWithState.Ptr.load();

    if (!ToBeDeleted)
      continue;

    auto KernIt = MKernelsPerProgramCache.find(ToBeDeleted);

    if (KernIt == MKernelsPerProgramCache.end())
      continue;

    for (auto &p : KernIt->second) {
      KernelWithBuildStateT &KernelWithState = p.second;
      PiKernelT *Kern = KernelWithState.Ptr.load();

      if (Kern)
        PI_CALL(piKernelRelease)(Kern);
    }

    PI_CALL(piProgramRelease)(ToBeDeleted);
  }
}
}
}
}

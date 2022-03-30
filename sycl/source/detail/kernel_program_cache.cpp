//==--- kernel_program_cache.cpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/kernel_program_cache.hpp>
#include <detail/plugin.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
KernelProgramCache::~KernelProgramCache() {
  for (auto &ProgIt : MCachedPrograms) {
    ProgramWithBuildStateT &ProgWithState = ProgIt.second;
    PiProgramT *ToBeDeleted = ProgWithState.Ptr.load();

    if (!ToBeDeleted)
      continue;

    auto KernIt = MKernelsPerProgramCache.find(ToBeDeleted);

    if (KernIt != MKernelsPerProgramCache.end()) {
      for (auto &p : KernIt->second) {
        KernelWithBuildStateT &KernelWithState = p.second;
        PiKernelT *Kern = KernelWithState.Ptr.load();

        if (Kern) {
          const detail::plugin &Plugin = MParentContext->getPlugin();
          Plugin.call<PiApiKind::piKernelRelease>(Kern);
        }
      }
      MKernelsPerProgramCache.erase(KernIt);
    }

    const detail::plugin &Plugin = MParentContext->getPlugin();
    Plugin.call<PiApiKind::piProgramRelease>(ToBeDeleted);
  }
}
}
}
}

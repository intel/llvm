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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
KernelProgramCache::~KernelProgramCache() {
  for (auto &ProgIt : MCachedPrograms.Cache) {
    ProgramWithBuildStateT &ProgWithState = ProgIt.second;
    RT::PiProgram *ToBeDeleted = ProgWithState.Ptr.load();

    if (!ToBeDeleted)
      continue;

    auto KernIt = MKernelsPerProgramCache.find(*ToBeDeleted);

    if (KernIt != MKernelsPerProgramCache.end()) {
      for (auto &p : KernIt->second) {
        BuildResult<KernelArgMaskPairT> &KernelWithState = p.second;
        KernelArgMaskPairT *KernelArgMaskPair = KernelWithState.Ptr.load();

        if (KernelArgMaskPair) {
          const PluginPtr &Plugin = MParentContext->getPlugin();
          Plugin->call<PiApiKind::piKernelRelease>(KernelArgMaskPair->first);
        }
      }
      MKernelsPerProgramCache.erase(KernIt);
    }

    const PluginPtr &Plugin = MParentContext->getPlugin();
    Plugin->call<PiApiKind::piProgramRelease>(*ToBeDeleted);
  }
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

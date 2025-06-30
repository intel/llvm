//==-------------------- kernel_name_based_cache_t.hpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/hashers.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <emhash/hash_table8.hpp>
#include <sycl/detail/kernel_name_str_t.hpp>
#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/ur.hpp>

#include <mutex>
#include <optional>

namespace sycl {
inline namespace _V1 {
namespace detail {
using FastKernelCacheKeyT = std::pair<ur_device_handle_t, ur_context_handle_t>;

struct FastKernelCacheVal {
  ur_kernel_handle_t MKernelHandle;    /* UR kernel handle pointer. */
  std::mutex *MMutex;                  /* Mutex guarding this kernel. When
                                     caching is disabled, the pointer is
                                     nullptr. */
  const KernelArgMask *MKernelArgMask; /* Eliminated kernel argument mask. */
  ur_program_handle_t MProgramHandle;  /* UR program handle corresponding to
                                     this kernel. */
  const adapter_impl &MAdapterPtr;     /* We can keep reference to the adapter
                                because during 2-stage shutdown the kernel
                                cache is destroyed deliberately before the
                                adapter. */

  FastKernelCacheVal(ur_kernel_handle_t KernelHandle, std::mutex *Mutex,
                     const KernelArgMask *KernelArgMask,
                     ur_program_handle_t ProgramHandle,
                     const adapter_impl &AdapterPtr)
      : MKernelHandle(KernelHandle), MMutex(Mutex),
        MKernelArgMask(KernelArgMask), MProgramHandle(ProgramHandle),
        MAdapterPtr(AdapterPtr) {}

  ~FastKernelCacheVal() {
    if (MKernelHandle)
      MAdapterPtr.call<sycl::detail::UrApiKind::urKernelRelease>(MKernelHandle);
    if (MProgramHandle)
      MAdapterPtr.call<sycl::detail::UrApiKind::urProgramRelease>(
          MProgramHandle);
    MKernelHandle = nullptr;
    MMutex = nullptr;
    MKernelArgMask = nullptr;
    MProgramHandle = nullptr;
  }

  FastKernelCacheVal(const FastKernelCacheVal &) = delete;
  FastKernelCacheVal &operator=(const FastKernelCacheVal &) = delete;
};
using FastKernelCacheValPtr = std::shared_ptr<FastKernelCacheVal>;

using FastKernelSubcacheMutexT = SpinLock;
using FastKernelSubcacheReadLockT = std::lock_guard<FastKernelSubcacheMutexT>;
using FastKernelSubcacheWriteLockT = std::lock_guard<FastKernelSubcacheMutexT>;

struct FastKernelEntryT {
  FastKernelCacheKeyT Key;
  FastKernelCacheValPtr Value;

  FastKernelEntryT(FastKernelCacheKeyT Key, const FastKernelCacheValPtr &Value)
      : Key(Key), Value(Value) {}

  FastKernelEntryT(const FastKernelEntryT &) = default;
  FastKernelEntryT &operator=(const FastKernelEntryT &) = default;
  FastKernelEntryT(FastKernelEntryT &&) = default;
  FastKernelEntryT &operator=(FastKernelEntryT &&) = default;
};

using FastKernelSubcacheEntriesT = std::vector<FastKernelEntryT>;

struct FastKernelSubcacheT {
  FastKernelSubcacheEntriesT Entries;
  FastKernelSubcacheMutexT Mutex;
};

// This class is used for caching kernel name based information.
// Pointers to instances of this class are stored in header function templates
// as a static variable to avoid repeated runtime lookup overhead.
class KernelNameBasedCacheT {
public:
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  KernelNameBasedCacheT() = default;
#endif
  KernelNameBasedCacheT(KernelNameStrRefT KernelName);

  void init(KernelNameStrRefT KernelName);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void initIfNeeded(KernelNameStrRefT KernelName);
#endif
  FastKernelSubcacheT &getKernelSubcache();
  bool usesAssert();
  const std::optional<int> &getImplicitLocalArgPos();

private:
  void assertInitialized();

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  std::atomic<bool> MInitialized = false;
#endif
  FastKernelSubcacheT MFastKernelSubcache;
  bool MUsesAssert;
  std::optional<int> MImplicitLocalArgPos;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

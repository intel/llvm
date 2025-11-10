//==---------------------- device_kernel_info.hpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/hashers.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <hash_table8.hpp>
#include <sycl/detail/compile_time_kernel_info.hpp>
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
  Managed<ur_kernel_handle_t> MKernelHandle; /* UR kernel. */
  std::mutex *MMutex;                        /* Mutex guarding this kernel. When
                                           caching is disabled, the pointer is
                                           nullptr. */
  const KernelArgMask *MKernelArgMask; /* Eliminated kernel argument mask. */
  Managed<ur_program_handle_t> MProgramHandle; /* UR program handle
                                    corresponding to this kernel. */
  adapter_impl &MAdapter; /* We can keep reference to the adapter
                            because during 2-stage shutdown the kernel
                            cache is destroyed deliberately before the
                            adapter. */

  FastKernelCacheVal(Managed<ur_kernel_handle_t> &&KernelHandle,
                     std::mutex *Mutex, const KernelArgMask *KernelArgMask,
                     Managed<ur_program_handle_t> &&ProgramHandle,
                     adapter_impl &Adapter)
      : MKernelHandle(std::move(KernelHandle)), MMutex(Mutex),
        MKernelArgMask(KernelArgMask), MProgramHandle(std::move(ProgramHandle)),
        MAdapter(Adapter) {}

  ~FastKernelCacheVal() {
    MMutex = nullptr;
    MKernelArgMask = nullptr;
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

// Structure for caching built kernels with a specific name.
// Used by instances of the kernel program cache class (potentially multiple).
struct FastKernelSubcacheT {
  FastKernelSubcacheEntriesT Entries;
  FastKernelSubcacheMutexT Mutex;
};

// This class aggregates information specific to device kernels (i.e.
// information that is uniform between different submissions of the same
// kernel). Pointers to instances of this class are stored in header function
// templates as a static variable to avoid repeated runtime lookup overhead.
// TODO Currently this class duplicates information fetched from the program
// manager. Instead, we should merge all of this information
// into this structure and get rid of the other KernelName -> * maps.
class DeviceKernelInfo : public CompileTimeKernelInfoTy {
public:
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Needs to own the kernel name string in non-preview builds since we pass it
  // using a temporary string instead of a string view there.
  std::string Name;
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  //DeviceKernelInfo() = default;
#endif
  DeviceKernelInfo() = default;
  DeviceKernelInfo(const CompileTimeKernelInfoTy &Info);

  void init(KernelNameStrRefT KernelName);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Initialize default-created entry that has no data recorded:
  void initIfEmpty(const CompileTimeKernelInfoTy &Info);
#endif
  void setCompileTimeInfoIfNeeded(const CompileTimeKernelInfoTy &Info);

  FastKernelSubcacheT &getKernelSubcache();
  bool usesAssert();
  const std::optional<int> &getImplicitLocalArgPos();

private:
  void assertInitialized();
  bool isCompileTimeInfoSet() const;

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

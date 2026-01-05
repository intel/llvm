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
#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/kernel_bundle.hpp>

#include <mutex>
#include <optional>
#include <string_view>

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
class DeviceKernelInfo : public CompileTimeKernelInfoTy {
public:
  DeviceKernelInfo(const CompileTimeKernelInfoTy &Info,
                   std::optional<sycl::kernel_id> KernelID = std::nullopt);

  void init(std::string_view KernelName);
  void setCompileTimeInfoIfNeeded(const CompileTimeKernelInfoTy &Info);

  FastKernelSubcacheT &getKernelSubcache() { return MFastKernelSubcache; }

  const std::optional<int> &getImplicitLocalArgPos() const {
    return MImplicitLocalArgPos;
  }

  const sycl::kernel_id &getKernelID() const {
    // Expected to be called only for DeviceKernelInfo instances created by
    // program manager (as opposed to allocated by sycl::kernel with
    // origins other than SYCL offline compilation).
    assert(MKernelID);
    return *MKernelID;
  }

  // Implicit local argument position is used only for some backends, so this
  // function allows setting it as more images are added.
  void setImplicitLocalArgPos(int Pos);

private:
  bool isCompileTimeInfoSet() const { return KernelSize != 0; }

  FastKernelSubcacheT MFastKernelSubcache;
  std::optional<int> MImplicitLocalArgPos;
  const std::optional<sycl::kernel_id> MKernelID;
};

} // namespace detail
} // namespace _V1
} // namespace sycl

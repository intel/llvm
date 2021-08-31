//==--------- global_handler.hpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/spinlock.hpp>
#include <CL/sycl/detail/util.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
class platform_impl;
class Scheduler;
class ProgramManager;
class Sync;
class plugin;
class device_filter_list;
class XPTIRegistry;

using PlatformImplPtr = std::shared_ptr<platform_impl>;

/// Wrapper class for global data structures with non-trivial destructors.
///
/// As user code can call SYCL Runtime functions from destructor of global
/// objects, it is not safe for the runtime library to have global objects with
/// non-trivial destructors. Such destructors can be called any time after
/// exiting main, which may result in user application crashes. Instead,
/// complex global objects must be wrapped into GlobalHandler. Its instance
/// is stored on heap, and deallocated when the runtime library is being
/// unloaded.
///
/// There's no need to store trivial globals here, as no code for their
/// construction or destruction is generated anyway.
class GlobalHandler {
public:
  /// \return a reference to a GlobalHandler singleton instance. Memory for
  /// storing objects is allocated on first call. The reference is valid as long
  /// as runtime library is loaded (i.e. untill `DllMain` or
  /// `__attribute__((destructor))` is called).
  static GlobalHandler &instance();

  GlobalHandler(const GlobalHandler &) = delete;
  GlobalHandler(GlobalHandler &&) = delete;

  Scheduler &getScheduler();
  ProgramManager &getProgramManager();
  Sync &getSync();
  std::vector<PlatformImplPtr> &getPlatformCache();
  std::mutex &getPlatformMapMutex();
  std::mutex &getFilterMutex();
  std::vector<plugin> &getPlugins();
  device_filter_list &getDeviceFilterList(const std::string &InitValue);
  XPTIRegistry &getXPTIRegistry();
  std::mutex &getHandlerExtendedMembersMutex();

private:
  friend void shutdown();

  // Constructor and destructor are declared out-of-line to allow incomplete
  // types as template arguments to unique_ptr.
  GlobalHandler();
  ~GlobalHandler();

  template <typename T> struct InstWithLock {
    std::unique_ptr<T> Inst;
    SpinLock Lock;
  };

  template <typename T, typename... Types>
  T &getOrCreate(InstWithLock<T> &IWL, Types... Args);

  InstWithLock<Scheduler> MScheduler;
  InstWithLock<ProgramManager> MProgramManager;
  InstWithLock<Sync> MSync;
  InstWithLock<std::vector<PlatformImplPtr>> MPlatformCache;
  InstWithLock<std::mutex> MPlatformMapMutex;
  InstWithLock<std::mutex> MFilterMutex;
  InstWithLock<std::vector<plugin>> MPlugins;
  InstWithLock<device_filter_list> MDeviceFilterList;
  InstWithLock<XPTIRegistry> MXPTIRegistry;
  // The mutex for synchronizing accesses to handlers extended members
  InstWithLock<std::mutex> MHandlerExtendedMembersMutex;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

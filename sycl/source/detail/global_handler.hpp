//==--------- global_handler.hpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/util.hpp>

#include <deque>
#include <memory>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
namespace detail {
class platform_impl;
class context_impl;
class Scheduler;
class ProgramManager;
class Sync;
class adapter_impl;
class ods_target_list;
class XPTIRegistry;
class ThreadPool;
struct KernelNameBasedCacheT;

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

  /// \return true if the instance has not been deallocated yet.
  static bool isInstanceAlive();

  GlobalHandler(const GlobalHandler &) = delete;
  GlobalHandler(GlobalHandler &&) = delete;
  GlobalHandler &operator=(const GlobalHandler &) = delete;

  void registerSchedulerUsage(bool ModifyCounter = true);
  Scheduler &getScheduler();
  bool isSchedulerAlive() const;
  ProgramManager &getProgramManager();
  Sync &getSync();
  std::vector<std::shared_ptr<platform_impl>> &getPlatformCache();

  void clearPlatforms();

  std::unordered_map<platform_impl *, std::shared_ptr<context_impl>> &
  getPlatformToDefaultContextCache();

  std::mutex &getPlatformToDefaultContextCacheMutex();
  std::mutex &getPlatformMapMutex();
  std::mutex &getFilterMutex();
  std::vector<adapter_impl *> &getAdapters();
  ods_target_list &getOneapiDeviceSelectorTargets(const std::string &InitValue);
  XPTIRegistry &getXPTIRegistry();
  ThreadPool &getHostTaskThreadPool();
  KernelNameBasedCacheT *createKernelNameBasedCache();
  static void registerStaticVarShutdownHandler();

  bool isOkToDefer() const;
  void endDeferredRelease();
  void unloadAdapters();
  void releaseDefaultContexts();
  void drainThreadPool();
  void prepareSchedulerToRelease(bool Blocking);

  void InitXPTI();
  void TraceEventXPTI(const char *Message);

  // For testing purposes only
  void attachScheduler(Scheduler *Scheduler);

private:
#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *GSYCLCallEvent = nullptr;
#endif

  bool OkToDefer = true;

  friend void shutdown_early();
  friend void shutdown_late();
  friend class ObjectUsageCounter;
  static GlobalHandler *&getInstancePtr();
  static SpinLock MSyclGlobalHandlerProtector;

  // Constructor and destructor are declared out-of-line to allow incomplete
  // types as template arguments to unique_ptr.
  GlobalHandler();
  ~GlobalHandler();

  template <typename T> struct InstWithLock {
    std::unique_ptr<T> Inst;
    SpinLock Lock;
  };

  template <typename T, typename... Types>
  T &getOrCreate(InstWithLock<T> &IWL, Types &&...Args);

  InstWithLock<Scheduler> MScheduler;
  InstWithLock<ProgramManager> MProgramManager;
  InstWithLock<Sync> MSync;
  InstWithLock<std::vector<std::shared_ptr<platform_impl>>> MPlatformCache;
  InstWithLock<
      std::unordered_map<platform_impl *, std::shared_ptr<context_impl>>>
      MPlatformToDefaultContextCache;
  InstWithLock<std::mutex> MPlatformToDefaultContextCacheMutex;
  InstWithLock<std::mutex> MPlatformMapMutex;
  InstWithLock<std::mutex> MFilterMutex;
  InstWithLock<std::vector<adapter_impl *>> MAdapters;
  InstWithLock<ods_target_list> MOneapiDeviceSelectorTargets;
  InstWithLock<XPTIRegistry> MXPTIRegistry;
  // Thread pool for host task and event callbacks execution
  InstWithLock<ThreadPool> MHostTaskThreadPool;
  InstWithLock<std::deque<KernelNameBasedCacheT>> MKernelNameBasedCaches;
};
} // namespace detail
} // namespace _V1
} // namespace sycl

//==--------- global_handler.hpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/spinlock.hpp>
#include <CL/sycl/detail/util.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <memory>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
using PlatformImplPtr = std::shared_ptr<platform_impl>;

struct GlobalHandler;

/// Wrapper class for global data structures with non-trivial destructors.
///
/// As user code can call SYCL Runtime functions from destructor of global
/// objects, it is not safe for the runtime library to have global objects with
/// non-trivial destructors. Such destructors can be called any time after
/// exiting main, which may result into user application crashes. Instead,
/// complex global objects must be wrapped into GlobalHandler. Its instance
/// is stored on heap, and deallocated when the runtime library is being
/// unloaded.
///
/// There's no need to store trivial globals here, as no code for their
/// construction or destruction is generated anyway.
class GlobalHandler {
public:
  /// \return a reference to a GlobalHandler singletone instance. Memory for
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


private:
  friend void shutdown();
  GlobalHandler() = default;

  SpinLock &getFieldsLock() { return MFieldsLock; }

  SpinLock MFieldsLock;

  std::unique_ptr<Scheduler> MScheduler{};
  std::unique_ptr<ProgramManager> MProgramManager{};
  std::unique_ptr<Sync> MSync{};
  std::unique_ptr<std::vector<PlatformImplPtr>> MPlatformCache{};
  std::unique_ptr<std::mutex> MPlatformMapMutex{};
  std::unique_ptr<std::mutex> MFilterMutex{};
  std::unique_ptr<std::vector<plugin>> MPlugins{};
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

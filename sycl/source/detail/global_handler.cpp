//==--------- global_handler.cpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/device_filter.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/spinlock.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
using LockGuard = std::lock_guard<SpinLock>;

GlobalHandler::GlobalHandler() = default;
GlobalHandler::~GlobalHandler() = default;

GlobalHandler &GlobalHandler::instance() {
  static GlobalHandler *SyclGlobalObjectsHandler = new GlobalHandler();
  return *SyclGlobalObjectsHandler;
}

template <typename T, typename... Types>
T &GlobalHandler::getOrCreate(InstWithLock<T> &IWL, Types... Args) {
  const LockGuard Lock{IWL.Lock};

  if (!IWL.Inst)
    IWL.Inst = std::make_unique<T>(Args...);

  return *IWL.Inst;
}

Scheduler &GlobalHandler::getScheduler() { return getOrCreate(MScheduler); }

ProgramManager &GlobalHandler::getProgramManager() {
  return getOrCreate(MProgramManager);
}

Sync &GlobalHandler::getSync() { return getOrCreate(MSync); }

std::vector<PlatformImplPtr> &GlobalHandler::getPlatformCache() {
  return getOrCreate(MPlatformCache);
}

std::mutex &GlobalHandler::getPlatformMapMutex() {
  return getOrCreate(MPlatformMapMutex);
}

std::mutex &GlobalHandler::getFilterMutex() {
  return getOrCreate(MFilterMutex);
}
std::vector<plugin> &GlobalHandler::getPlugins() {
  return getOrCreate(MPlugins);
}
device_filter_list &
GlobalHandler::getDeviceFilterList(const std::string &InitValue) {
  return getOrCreate(MDeviceFilterList, InitValue);
}

XPTIRegistry &GlobalHandler::getXPTIRegistry() {
  return getOrCreate(MXPTIRegistry);
}

std::mutex &GlobalHandler::getHandlerExtendedMembersMutex() {
  return getOrCreate(MHandlerExtendedMembersMutex);
}

void shutdown() {
  // First, release resources, that may access plugins.
  GlobalHandler::instance().MScheduler.Inst.reset(nullptr);
  GlobalHandler::instance().MProgramManager.Inst.reset(nullptr);
  GlobalHandler::instance().MPlatformCache.Inst.reset(nullptr);

  // Call to GlobalHandler::instance().getPlugins() initializes plugins. If
  // user application has loaded SYCL runtime, and never called any APIs,
  // there's no need to load and unload plugins.
  if (GlobalHandler::instance().MPlugins.Inst) {
    for (plugin &Plugin : GlobalHandler::instance().getPlugins()) {
      // PluginParameter is reserved for future use that can control
      // some parameters in the plugin tear-down process.
      // Currently, it is not used.
      void *PluginParameter = nullptr;
      Plugin.call<PiApiKind::piTearDown>(PluginParameter);
      Plugin.unload();
    }
    GlobalHandler::instance().MPlugins.Inst.reset(nullptr);
  }

  // Release the rest of global resources.
  delete &GlobalHandler::instance();
}

#ifdef _WIN32
extern "C" __SYCL_EXPORT BOOL WINAPI DllMain(HINSTANCE hinstDLL,
                                             DWORD fdwReason,
                                             LPVOID lpReserved) {
  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_DETACH:
    shutdown();
    break;
  case DLL_PROCESS_ATTACH:
  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}
#else
// Setting low priority on destructor ensures it runs after all other global
// destructors. Priorities 0-100 are reserved by the compiler. The priority
// value 110 allows SYCL users to run their destructors after runtime library
// deinitialization.
__attribute__((destructor(110))) static void syclUnload() { shutdown(); }
#endif
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

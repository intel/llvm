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

#ifdef _WIN32
#include <windows.h>
#endif

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
GlobalHandler::GlobalHandler() = default;
GlobalHandler::~GlobalHandler() = default;

GlobalHandler &GlobalHandler::instance() {
  static GlobalHandler *SyclGlobalObjectsHandler = new GlobalHandler();
  return *SyclGlobalObjectsHandler;
}

Scheduler &GlobalHandler::getScheduler() {
  if (MScheduler)
    return *MScheduler;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MScheduler)
    MScheduler = std::make_unique<Scheduler>();

  return *MScheduler;
}
ProgramManager &GlobalHandler::getProgramManager() {
  if (MProgramManager)
    return *MProgramManager;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MProgramManager)
    MProgramManager = std::make_unique<ProgramManager>();

  return *MProgramManager;
}
Sync &GlobalHandler::getSync() {
  if (MSync)
    return *MSync;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MSync)
    MSync = std::make_unique<Sync>();

  return *MSync;
}
std::vector<PlatformImplPtr> &GlobalHandler::getPlatformCache() {
  if (MPlatformCache)
    return *MPlatformCache;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MPlatformCache)
    MPlatformCache = std::make_unique<std::vector<PlatformImplPtr>>();

  return *MPlatformCache;
}
std::mutex &GlobalHandler::getPlatformMapMutex() {
  if (MPlatformMapMutex)
    return *MPlatformMapMutex;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MPlatformMapMutex)
    MPlatformMapMutex = std::make_unique<std::mutex>();

  return *MPlatformMapMutex;
}
std::mutex &GlobalHandler::getFilterMutex() {
  if (MFilterMutex)
    return *MFilterMutex;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MFilterMutex)
    MFilterMutex = std::make_unique<std::mutex>();

  return *MFilterMutex;
}
std::vector<plugin> &GlobalHandler::getPlugins() {
  if (MPlugins)
    return *MPlugins;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MPlugins)
    MPlugins = std::make_unique<std::vector<plugin>>();

  return *MPlugins;
}
device_filter_list &
GlobalHandler::getDeviceFilterList(const std::string &InitValue) {
  if (MDeviceFilterList)
    return *MDeviceFilterList;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MDeviceFilterList)
    MDeviceFilterList = std::make_unique<device_filter_list>(InitValue);

  return *MDeviceFilterList;
}

std::mutex &GlobalHandler::getHandlerExtendedMembersMutex() {
  if (MHandlerExtendedMembersMutex)
    return *MHandlerExtendedMembersMutex;

  const std::lock_guard<SpinLock> Lock{MFieldsLock};
  if (!MHandlerExtendedMembersMutex)
    MHandlerExtendedMembersMutex = std::make_unique<std::mutex>();

  return *MHandlerExtendedMembersMutex;
}

void shutdown() {
  // First, release resources, that may access plugins.
  GlobalHandler::instance().MScheduler.reset(nullptr);
  GlobalHandler::instance().MProgramManager.reset(nullptr);
  GlobalHandler::instance().MPlatformCache.reset(nullptr);

  // Call to GlobalHandler::instance().getPlugins() initializes plugins. If
  // user application has loaded SYCL runtime, and never called any APIs,
  // there's no need to load and unload plugins.
  if (GlobalHandler::instance().MPlugins) {
    for (plugin &Plugin : GlobalHandler::instance().getPlugins()) {
      // PluginParameter is reserved for future use that can control
      // some parameters in the plugin tear-down process.
      // Currently, it is not used.
      void *PluginParameter = nullptr;
      Plugin.call_nocheck<PiApiKind::piTearDown>(PluginParameter);
      Plugin.unload();
    }
    GlobalHandler::instance().MPlugins.reset(nullptr);
  }

  // Release the rest of global resources.
  delete &GlobalHandler::instance();
}

#ifdef _WIN32
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
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

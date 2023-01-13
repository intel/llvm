//==--------- global_handler.cpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/thread_pool.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/spinlock.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

using LockGuard = std::lock_guard<SpinLock>;
SpinLock GlobalHandler::MSyclGlobalHandlerProtector{};

// Utility class to track references on object.
// Used for GlobalHandler now and created as thread_local object on the first
// Scheduler usage. Origin idea is to track usage of Scheduler from main and
// other used threads - they increment MCounter; and to use but not add extra
// reference by our thread_pool threads. For this control MIncrementCounter
// class member is used.
class ObjectUsageCounter {
public:
  ObjectUsageCounter(bool ModifyCounter) : MModifyCounter(ModifyCounter) {
    if (MModifyCounter)
      MCounter++;
  }
  ~ObjectUsageCounter() {
    if (!MModifyCounter)
      return;

    MCounter--;
    if (!MCounter) {
      LockGuard Guard(GlobalHandler::MSyclGlobalHandlerProtector);
      GlobalHandler *RTGlobalObjHandler = GlobalHandler::getInstancePtr();
      if (RTGlobalObjHandler) {
        RTGlobalObjHandler->prepareSchedulerToRelease();
      }
    }
  }

private:
  static std::atomic_uint MCounter;
  bool MModifyCounter;
};
std::atomic_uint ObjectUsageCounter::MCounter{0};

GlobalHandler::GlobalHandler() = default;
GlobalHandler::~GlobalHandler() = default;

GlobalHandler *&GlobalHandler::getInstancePtr() {
  static GlobalHandler *RTGlobalObjHandler = new GlobalHandler();
  return RTGlobalObjHandler;
}

GlobalHandler &GlobalHandler::instance() {
  GlobalHandler *RTGlobalObjHandler = GlobalHandler::getInstancePtr();
  assert(RTGlobalObjHandler && "Handler must not be deallocated earlier");
  return *RTGlobalObjHandler;
}

template <typename T, typename... Types>
T &GlobalHandler::getOrCreate(InstWithLock<T> &IWL, Types... Args) {
  const LockGuard Lock{IWL.Lock};

  if (!IWL.Inst)
    IWL.Inst = std::make_unique<T>(Args...);

  return *IWL.Inst;
}

void GlobalHandler::attachScheduler(Scheduler *Scheduler) {
  // The method is used in unit tests only. Do not protect with lock since
  // releaseResources will cause dead lock due to host queue release
  if (MScheduler.Inst)
    prepareSchedulerToRelease();
  MScheduler.Inst.reset(Scheduler);
}

Scheduler &GlobalHandler::getScheduler() {
  getOrCreate(MScheduler);
  registerSchedulerUsage();
  return *MScheduler.Inst;
}

void GlobalHandler::registerSchedulerUsage(bool ModifyCounter) {
  thread_local ObjectUsageCounter SchedulerCounter(ModifyCounter);
}

ProgramManager &GlobalHandler::getProgramManager() {
  return getOrCreate(MProgramManager);
}

std::unordered_map<PlatformImplPtr, ContextImplPtr> &
GlobalHandler::getPlatformToDefaultContextCache() {
  return getOrCreate(MPlatformToDefaultContextCache);
}

std::mutex &GlobalHandler::getPlatformToDefaultContextCacheMutex() {
  return getOrCreate(MPlatformToDefaultContextCacheMutex);
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

ods_target_list &
GlobalHandler::getOneapiDeviceSelectorTargets(const std::string &InitValue) {
  return getOrCreate(MOneapiDeviceSelectorTargets, InitValue);
}

XPTIRegistry &GlobalHandler::getXPTIRegistry() {
  return getOrCreate(MXPTIRegistry);
}

ThreadPool &GlobalHandler::getHostTaskThreadPool() {
  int Size = SYCLConfig<SYCL_QUEUE_THREAD_POOL_SIZE>::get();
  ThreadPool &TP = getOrCreate(MHostTaskThreadPool, Size);

  return TP;
}

void GlobalHandler::releaseDefaultContexts() {
  // Release shared-pointers to SYCL objects.
#ifndef _WIN32
  MPlatformToDefaultContextCache.Inst.reset(nullptr);
#else
  // Windows does not maintain dependencies between dynamically loaded libraries
  // and can unload SYCL runtime dependencies before sycl.dll's DllMain has
  // finished. To avoid calls to nowhere, intentionally leak platform to device
  // cache. This will prevent destructors from being called, thus no PI cleanup
  // routines will be called in the end.
  // Update: the win_proxy_loader addresses this for SYCL's own dependencies,
  // but the GPU device dlls seem to manually load yet another DLL which may
  // have been released when this function is called. So we still release() and
  // leak until that is addressed. DefaultContext destructs fine on CPU device.
  MPlatformToDefaultContextCache.Inst.release();
#endif
}


// Note: Split from shutdown so it is available to the unittests for ensuring
//       that the mock plugin is the lone plugin.
void GlobalHandler::unloadPlugins() {
  // Call to GlobalHandler::instance().getPlugins() initializes plugins. If
  // user application has loaded SYCL runtime, and never called any APIs,
  // there's no need to load and unload plugins.
  if (MPlugins.Inst) {
    for (plugin &Plugin : getPlugins()) {
      // PluginParameter is reserved for future use that can control
      // some parameters in the plugin tear-down process.
      // Currently, it is not used.
      void *PluginParameter = nullptr;
      Plugin.call<PiApiKind::piTearDown>(PluginParameter);
      Plugin.unload();
    }
  }
  // Clear after unload to avoid uses after unload.
  getPlugins().clear();
}

void GlobalHandler::prepareSchedulerToRelease() {
#ifdef __SYCL_DEFER_MEM_OBJ_DESTRUCTION
  drainThreadPool();
#endif
  if (MScheduler.Inst)
    MScheduler.Inst->releaseResources();
}

void GlobalHandler::drainThreadPool() {
  if (MHostTaskThreadPool.Inst)
    MHostTaskThreadPool.Inst->drain();
}

void shutdown() {
  const LockGuard Lock{GlobalHandler::MSyclGlobalHandlerProtector};
  GlobalHandler *&Handler = GlobalHandler::getInstancePtr();
  if (!Handler)
    return;

  // Ensure neither host task is working so that no default context is accessed
  // upon its release
  Handler->prepareSchedulerToRelease();

  if (Handler->MHostTaskThreadPool.Inst)
    Handler->MHostTaskThreadPool.Inst->finishAndWait();

  // If default contexts are requested after the first default contexts have
  // been released there may be a new default context. These must be released
  // prior to closing the plugins.
  // Note: Releasing a default context here may cause failures in plugins with
  // global state as the global state may have been released.
  Handler->releaseDefaultContexts();

  // First, release resources, that may access plugins.
  Handler->MPlatformCache.Inst.reset(nullptr);
  Handler->MScheduler.Inst.reset(nullptr);
  Handler->MProgramManager.Inst.reset(nullptr);

  // Clear the plugins and reset the instance if it was there.
  Handler->unloadPlugins();
  if (Handler->MPlugins.Inst)
    Handler->MPlugins.Inst.reset(nullptr);

  // Release the rest of global resources.
  delete Handler;
  Handler = nullptr;
}

#ifdef _WIN32
extern "C" __SYCL_EXPORT BOOL WINAPI DllMain(HINSTANCE hinstDLL,
                                             DWORD fdwReason,
                                             LPVOID lpReserved) {
  bool PrintPiTrace = false;
  static const char *PiTrace = std::getenv("SYCL_PI_TRACE");
  static const int PiTraceValue = PiTrace ? std::stoi(PiTrace) : 0;
  if (PiTraceValue == -1 || PiTraceValue == 2) { // Means print all PI traces
    PrintPiTrace = true;
  }

  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_DETACH:
    if (PrintPiTrace)
      std::cout << "---> DLL_PROCESS_DETACH syclx.dll\n" << std::endl;

#ifdef XPTI_ENABLE_INSTRUMENTATION
    if (xptiTraceEnabled())
      return TRUE; // When doing xpti tracing, we can't safely call shutdown.
    // TODO: figure out what XPTI is doing that prevents release.
#endif

    shutdown();
    break;
  case DLL_PROCESS_ATTACH:
    if (PrintPiTrace)
      std::cout << "---> DLL_PROCESS_ATTACH syclx.dll\n" << std::endl;
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

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

// Utility class to track references on object.
// Used for Scheduler now and created as thread_local object.
// Origin idea is to track usage of Scheduler from main and other used threads -
// they increment MCounter; and to use but not add extra reference by our
// thread_pool threads. For this control MIncrementCounter class member is used.
template <class ResourceHandler> class ObjectUsageCounter {
public:
  // Note: -Wctad-maybe-unsupported may generate warning if no ResourceHandler
  // type explicitly declared.
  ObjectUsageCounter(std::unique_ptr<ResourceHandler> &Obj, bool ModifyCounter)
      : MModifyCounter(ModifyCounter), MObj(Obj) {
    if (MModifyCounter)
      MCounter++;
  }
  ~ObjectUsageCounter() {
    if (!MModifyCounter)
      return;

    MCounter--;
    if (!MCounter && MObj)
      MObj->releaseResources();
  }

private:
  static std::atomic_uint MCounter;
  bool MModifyCounter;
  std::unique_ptr<ResourceHandler> &MObj;
};
template <class ResourceHandler>
std::atomic_uint ObjectUsageCounter<ResourceHandler>::MCounter{0};

using LockGuard = std::lock_guard<SpinLock>;

GlobalHandler::GlobalHandler() = default;
GlobalHandler::~GlobalHandler() = default;

void GlobalHandler::InitXPTIStuff() {
  // Let subscribers know a new stream is being initialized
  getXPTIRegistry().initializeStream(SYCL_SYCLCALL_STREAM_NAME, GMajVer,
                                     GMinVer, GVerStr);
  xpti::payload_t SYCLPayload("SYCL Interface Layer");
  uint64_t SYCLInstanceNo;
  GSYCLCallEvent =
      xptiMakeEvent("SYCL API Layer", &SYCLPayload, xpti::trace_algorithm_event,
                    xpti_at::active, &SYCLInstanceNo);
}

std::string BuildPayloadStr(const code_location &Payload,
                            const std::shared_ptr<std::string> &Message) {
  std::string Result;
  const char Separator[] = ":";
  auto FileName = Payload.fileName();
  auto FunctionName = Payload.functionName();
  if (FileName) {
    Result.append(FileName);
    Result.append(Separator);
  }
  if (FunctionName) {
    Result.append(FunctionName);
    Result.append(Separator);
    Result.append("ln");
    Result += std::to_string(Payload.lineNumber());
    Result.append(Separator);
    Result.append("col");
    Result += std::to_string(Payload.columnNumber());
  }
  if (Result.empty()) {
    Result = "unknown";
  }
  if (!Message->empty()) {
    Result += ":" + *Message;
  }
  return Result;
}

void GlobalHandler::TraceEventXPTI(
    const std::shared_ptr<std::string> &Message) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (xptiTraceEnabled()) {
    uint64_t Uid = xptiGetUniqueId(); // Gets the UID from TLS
    uint8_t StreamID = xptiRegisterStream(SYCL_SYCLCALL_STREAM_NAME);
    detail::tls_code_loc_t Tls;
    auto TData = Tls.query();
    xptiNotifySubscribers(
        StreamID, (uint16_t)xpti::trace_point_type_t::diagnostics,
        GSYCLCallEvent, nullptr, Uid,
        static_cast<const void *>(BuildPayloadStr(TData, Message).c_str()));
  }

#endif
}

GlobalHandler &GlobalHandler::instance() {
  static GlobalHandler *SyclGlobalObjectsHandler = new GlobalHandler();

#ifdef XPTI_ENABLE_INSTRUMENTATION
  static std::once_flag InitXPTI;
  if (xptiTraceEnabled()) {
    std::call_once(InitXPTI,
                   [&]() { SyclGlobalObjectsHandler->InitXPTIStuff(); });
  }
#endif
  return *SyclGlobalObjectsHandler;
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
    MScheduler.Inst->releaseResources();
  MScheduler.Inst.reset(Scheduler);
}

Scheduler &GlobalHandler::getScheduler() {
  getOrCreate(MScheduler);
  registerSchedulerUsage();
  return *MScheduler.Inst;
}

void GlobalHandler::registerSchedulerUsage(bool ModifyCounter) {
  thread_local ObjectUsageCounter<Scheduler> SchedulerCounter(MScheduler.Inst,
                                                              ModifyCounter);
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
  GlobalHandler::instance().MPlatformToDefaultContextCache.Inst.reset(nullptr);
#else
  // Windows does not maintain dependencies between dynamically loaded libraries
  // and can unload SYCL runtime dependencies before sycl.dll's DllMain has
  // finished. To avoid calls to nowhere, intentionally leak platform to device
  // cache. This will prevent destructors from being called, thus no PI cleanup
  // routines will be called in the end.
  GlobalHandler::instance().MPlatformToDefaultContextCache.Inst.release();
#endif
}

struct DefaultContextReleaseHandler {
  ~DefaultContextReleaseHandler() {
    GlobalHandler::instance().releaseDefaultContexts();
  }
};

void GlobalHandler::registerDefaultContextReleaseHandler() {
  static DefaultContextReleaseHandler handler{};
}

// Note: Split from shutdown so it is available to the unittests for ensuring
//       that the mock plugin is the lone plugin.
void GlobalHandler::unloadPlugins() {
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
  }
  // Clear after unload to avoid uses after unload.
  GlobalHandler::instance().getPlugins().clear();
}

void GlobalHandler::drainThreadPool() {
  if (MHostTaskThreadPool.Inst)
    MHostTaskThreadPool.Inst->drain();
}

void shutdown() {
  // Ensure neither host task is working so that no default context is accessed
  // upon its release

  if (GlobalHandler::instance().MScheduler.Inst)
    GlobalHandler::instance().MScheduler.Inst->releaseResources();

  if (GlobalHandler::instance().MHostTaskThreadPool.Inst)
    GlobalHandler::instance().MHostTaskThreadPool.Inst->finishAndWait();

  // If default contexts are requested after the first default contexts have
  // been released there may be a new default context. These must be released
  // prior to closing the plugins.
  // Note: Releasing a default context here may cause failures in plugins with
  // global state as the global state may have been released.
  GlobalHandler::instance().releaseDefaultContexts();

  // First, release resources, that may access plugins.
  GlobalHandler::instance().MPlatformCache.Inst.reset(nullptr);
  GlobalHandler::instance().MScheduler.Inst.reset(nullptr);
  GlobalHandler::instance().MProgramManager.Inst.reset(nullptr);

  // Clear the plugins and reset the instance if it was there.
  GlobalHandler::instance().unloadPlugins();
  if (GlobalHandler::instance().MPlugins.Inst)
    GlobalHandler::instance().MPlugins.Inst.reset(nullptr);

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
    if (!lpReserved)
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

//==--------- global_handler.cpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef ENABLE_STACK_TRACE
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"
#endif

#include <detail/adapter_impl.hpp>
#include <detail/config.hpp>
#include <detail/device_kernel_info.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/thread_pool.hpp>
#include <detail/ur.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/detail/device_filter.hpp>
#include <sycl/detail/spinlock.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

using LockGuard = std::lock_guard<SpinLock>;
SpinLock GlobalHandler::MSyclGlobalHandlerProtector{};

// forward decl
void shutdown_early(bool);
void shutdown_late();
#ifdef _WIN32
BOOL isLinkedStatically();
#endif

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
    try {
      if (!MModifyCounter)
        return;

      LockGuard Guard(GlobalHandler::MSyclGlobalHandlerProtector);
      MCounter--;
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~ObjectUsageCounter", e);
    }
  }

private:
  static std::atomic_uint MCounter;
  bool MModifyCounter;
};
std::atomic_uint ObjectUsageCounter::MCounter{0};

GlobalHandler::GlobalHandler() = default;
GlobalHandler::~GlobalHandler() = default;

void GlobalHandler::TraceEventXPTI(const char *Message) {
  if (!Message)
    return;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  // This section is used to emit XPTI trace events when exceptions occur
  if (xptiTraceEnabled()) {
    GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();

    // We have to handle the cases where: (1) we may have just the code location
    // set and not UID and (2) UID set
    detail::tls_code_loc_t Tls;
    auto CodeLocation = Tls.query();

    // Creating a tracepoint using the stashed code location and notifying the
    // subscriber with the diagnostic message
    xpti::framework::tracepoint_scope_t TP(
        CodeLocation.fileName(), CodeLocation.functionName(),
        CodeLocation.lineNumber(), CodeLocation.columnNumber(), nullptr);
    // Notify the subscriber with a diagnostic message when an exception occurs.
    TP.stream(detail::getActiveXPTIStreamID())
        .traceType(xpti::trace_point_type_t::diagnostics)
        .parentEvent(GSYCLCallEvent)
        .notify(static_cast<const void *>(Message));
  }

#endif
}

GlobalHandler *GlobalHandler::RTGlobalObjHandler = new GlobalHandler();

template <typename T, typename... Types>
T &GlobalHandler::getOrCreate(InstWithLock<T> &IWL, Types &&...Args) {
  const LockGuard Lock{IWL.Lock};

  if (!IWL.Inst)
    IWL.Inst = std::make_unique<T>(std::forward<Types>(Args)...);

  return *IWL.Inst;
}

void GlobalHandler::attachScheduler(Scheduler *Scheduler) {
  // The method is used in unit tests only. Do not protect with lock since
  // releaseResources will cause dead lock due to host queue release
  if (MScheduler.Inst)
    prepareSchedulerToRelease(true);
  MScheduler.Inst.reset(Scheduler);
}

static void enableOnCrashStackPrinting() {
#ifdef ENABLE_STACK_TRACE
  static std::once_flag PrintStackFlag;
  std::call_once(PrintStackFlag, []() {
    llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  });
#endif
}

Scheduler &GlobalHandler::getScheduler() {
  getOrCreate(MScheduler);
  registerSchedulerUsage();
  // On Windows the registration of the signal handler before main function
  // (e.g. from DLLMain or from constructors of program scope objects) doesn't
  // work. So, registering signal handler here because:
  // 1) getScheduler is likely to be called for any non-trivial application;
  // 2) first call to getScheduler is likely to be done after main starts.
  // The same is done in getAdapters.
  enableOnCrashStackPrinting();
  return *MScheduler.Inst;
}

bool GlobalHandler::isSchedulerAlive() const { return MScheduler.Inst.get(); }

void GlobalHandler::registerSchedulerUsage(bool ModifyCounter) {
  thread_local ObjectUsageCounter SchedulerCounter(ModifyCounter);
}

ProgramManager &GlobalHandler::getProgramManager() {
  static ProgramManager &PM = getOrCreate(MProgramManager);
  return PM;
}

std::unordered_map<platform_impl *, std::shared_ptr<context_impl>> &
GlobalHandler::getPlatformToDefaultContextCache() {
  // The optimization with static reference is not done because
  // there are public methods of the GlobalHandler
  // that can set the MPlatformToDefaultContextCache back to nullptr.
  // So one time initialization is not possible and we need
  // to call getOrCreate on every access.
  return getOrCreate(MPlatformToDefaultContextCache);
}

std::mutex &GlobalHandler::getPlatformToDefaultContextCacheMutex() {
  static std::mutex &PlatformToDefaultContextCacheMutex =
      getOrCreate(MPlatformToDefaultContextCacheMutex);
  return PlatformToDefaultContextCacheMutex;
}

Sync &GlobalHandler::getSync() {
  static Sync &sync = getOrCreate(MSync);
  return sync;
}

std::vector<std::shared_ptr<platform_impl>> &GlobalHandler::getPlatformCache() {
  static std::vector<std::shared_ptr<platform_impl>> &PlatformCache =
      getOrCreate(MPlatformCache);
  return PlatformCache;
}

std::mutex &GlobalHandler::getPlatformMapMutex() {
  static std::mutex &PlatformMapMutex = getOrCreate(MPlatformMapMutex);
  return PlatformMapMutex;
}

std::mutex &GlobalHandler::getFilterMutex() {
  static std::mutex &FilterMutex = getOrCreate(MFilterMutex);
  return FilterMutex;
}

std::vector<adapter_impl *> &GlobalHandler::getAdapters() {
  static std::vector<adapter_impl *> &adapters = getOrCreate(MAdapters);
  enableOnCrashStackPrinting();
  return adapters;
}

ods_target_list &
GlobalHandler::getOneapiDeviceSelectorTargets(const std::string &InitValue) {
  static ods_target_list &OneapiDeviceSelectorTargets =
      getOrCreate(MOneapiDeviceSelectorTargets, InitValue);
  return OneapiDeviceSelectorTargets;
}

XPTIRegistry &GlobalHandler::getXPTIRegistry() {
  return getOrCreate(MXPTIRegistry);
}

ThreadPool &GlobalHandler::getHostTaskThreadPool() {
  static ThreadPool &TP = getOrCreate(
      MHostTaskThreadPool, SYCLConfig<SYCL_QUEUE_THREAD_POOL_SIZE>::get());
  return TP;
}

void GlobalHandler::releaseDefaultContexts() {
  // Release shared-pointers to SYCL objects.
  // Note that on Windows the destruction of the default context
  // races with the detaching of the DLL object that calls urLoaderTearDown.

  MPlatformToDefaultContextCache.Inst.reset(nullptr);
}

// Shutdown is split into two parts. shutdown_early() stops any more
// objects from being deferred and takes an initial pass at freeing them.
// shutdown_late() finishes and releases the adapters and the GlobalHandler.
// For Windows, early shutdown is typically called from DllMain,
// and late shutdown is here.
// For Linux, early shutdown is here, and late shutdown is called from
// a low priority destructor.
struct StaticVarShutdownHandler {
  StaticVarShutdownHandler(const StaticVarShutdownHandler &) = delete;
  StaticVarShutdownHandler &
  operator=(const StaticVarShutdownHandler &) = delete;
  ~StaticVarShutdownHandler() {
    try {
#ifdef _WIN32
      // If statically linked, DllMain will not be called. So we do its work
      // here.
      if (isLinkedStatically()) {
        shutdown_early(true);
      }

      shutdown_late();
#else
      shutdown_early(true);
#endif
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM(
          "exception in ~StaticVarShutdownHandler", e);
    }
  }
};

void GlobalHandler::registerStaticVarShutdownHandler() {
  static StaticVarShutdownHandler handler{};
}

bool GlobalHandler::isOkToDefer() const { return OkToDefer; }

void GlobalHandler::endDeferredRelease() { OkToDefer = false; }

// Note: Split from shutdown so it is available to the unittests for ensuring
//       that the mock adapter is the lone adapter.
void GlobalHandler::unloadAdapters() {
  // Call to GlobalHandler::instance().getAdapters() initializes adapters. If
  // user application has loaded SYCL runtime, and never called any APIs,
  // there's no need to load and unload adapters.
  if (MAdapters.Inst) {
    for (const auto &Adapter : getAdapters()) {
      Adapter->release();
      delete Adapter;
    }
  }

  UrFuncInfo<UrApiKind::urLoaderTearDown> loaderTearDownInfo;
  auto loaderTearDown =
      loaderTearDownInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
  loaderTearDown();
  // urLoaderTearDown();

  // Clear after unload to avoid uses after unload.
  getAdapters().clear();
}

void GlobalHandler::prepareSchedulerToRelease(bool Blocking) {
#ifndef _WIN32
  if (Blocking)
    drainThreadPool();
#endif
  if (MScheduler.Inst)
    MScheduler.Inst->releaseResources(Blocking ? BlockingT::BLOCKING
                                               : BlockingT::NON_BLOCKING);
}

void GlobalHandler::drainThreadPool() {
  if (MHostTaskThreadPool.Inst)
    MHostTaskThreadPool.Inst->drain();
}

// Note: this function can be called on Windows twice:
//  1) when library is unloaded via FreeLibrary
//  2) when process is being terminated
void shutdown_early(bool CanJoinThreads = true) {
  const LockGuard Lock{GlobalHandler::MSyclGlobalHandlerProtector};
  if (!GlobalHandler::RTGlobalObjHandler)
    return;

#if defined(XPTI_ENABLE_INSTRUMENTATION) && defined(_WIN32)
  if (xptiTraceEnabled())
    return; // When doing xpti tracing, we can't safely shutdown on Win.
            // TODO: figure out why XPTI prevents release.
#endif

  // Now that we are shutting down, we will no longer defer MemObj releases.
  GlobalHandler::RTGlobalObjHandler->endDeferredRelease();

  // Ensure neither host task is working so that no default context is accessed
  // upon its release
  GlobalHandler::RTGlobalObjHandler->prepareSchedulerToRelease(true);

  if (GlobalHandler::RTGlobalObjHandler->MHostTaskThreadPool.Inst) {
    GlobalHandler::RTGlobalObjHandler->MHostTaskThreadPool.Inst->finishAndWait(
        CanJoinThreads);
    GlobalHandler::RTGlobalObjHandler->MHostTaskThreadPool.Inst.reset(nullptr);
  }

  // This releases OUR reference to the default context, but
  // other may yet have refs
  GlobalHandler::RTGlobalObjHandler->releaseDefaultContexts();
}

void shutdown_late() {
  const LockGuard Lock{GlobalHandler::MSyclGlobalHandlerProtector};
  if (!GlobalHandler::RTGlobalObjHandler)
    return;

#if defined(XPTI_ENABLE_INSTRUMENTATION) && defined(_WIN32)
  if (xptiTraceEnabled())
    return; // When doing xpti tracing, we can't safely shutdown on Win.
            // TODO: figure out why XPTI prevents release.
#endif

  // First, release resources, that may access adapters.
  GlobalHandler::RTGlobalObjHandler->MPlatformCache.Inst.reset(nullptr);
  GlobalHandler::RTGlobalObjHandler->MScheduler.Inst.reset(nullptr);
  GlobalHandler::RTGlobalObjHandler->MProgramManager.Inst.reset(nullptr);

  // Clear the adapters and reset the instance if it was there.
  GlobalHandler::RTGlobalObjHandler->unloadAdapters();
  if (GlobalHandler::RTGlobalObjHandler->MAdapters.Inst)
    GlobalHandler::RTGlobalObjHandler->MAdapters.Inst.reset(nullptr);

  GlobalHandler::RTGlobalObjHandler->MXPTIRegistry.Inst.reset(nullptr);

  // Release the rest of global resources.
  delete GlobalHandler::RTGlobalObjHandler;
  GlobalHandler::RTGlobalObjHandler = nullptr;
}

#ifdef _WIN32
extern "C" __SYCL_EXPORT BOOL WINAPI DllMain(HINSTANCE hinstDLL,
                                             DWORD fdwReason,
                                             LPVOID lpReserved) {
  bool PrintUrTrace = false;
  try {
    PrintUrTrace =
        sycl::detail::ur::trace(sycl::detail::ur::TraceLevel::TRACE_CALLS);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in DllMain", e);
    return FALSE;
  }

  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_DETACH:
    if (PrintUrTrace)
      std::cout << "---> DLL_PROCESS_DETACH syclx.dll\n" << std::endl;

    try {
      // WA for threads handling. We must call join() or detach() on host task
      // execution thread to avoid UB. lpReserved == NULL if library is unloaded
      // via FreeLibrary. In this case we can't join threads within DllMain call
      // due to global loader lock and DLL_THREAD_DETACH signalling. lpReserved
      // != NULL if library is unloaded during process termination. In this case
      // Windows terminates threads but leave them in signalled state, prevents
      // DLL_THREAD_DETACH notification and we can call join() as NOP.
      shutdown_early(lpReserved != NULL);
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in DLL_PROCESS_DETACH", e);
      return FALSE;
    }

    break;
  case DLL_PROCESS_ATTACH:
    if (PrintUrTrace)
      std::cout << "---> DLL_PROCESS_ATTACH syclx.dll\n" << std::endl;

    break;
  case DLL_THREAD_ATTACH:
    break;
  case DLL_THREAD_DETACH:
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}
BOOL isLinkedStatically() {
  // If the exePath is the same as the dllPath,
  // or if the module handle for DllMain is not retrievable,
  // then we are linked statically
  // Otherwise we are dynamically linked or loaded.
  HMODULE hModule = nullptr;
  auto LpModuleAddr = reinterpret_cast<LPCSTR>(&DllMain);
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, LpModuleAddr,
                          &hModule)) {
    return true; // not retrievable, therefore statically linked
  }
  char dllPath[MAX_PATH];
  if (GetModuleFileNameA(hModule, dllPath, MAX_PATH)) {
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
      if (std::string(dllPath) == std::string(exePath)) {
        return true; // paths identical, therefore statically linked
      }
    }
  }
  return false; // Otherwise dynamically linked or loaded
}
#else
// Setting low priority on destructor ensures it runs after all other global
// destructors. Priorities 0-100 are reserved by the compiler. The priority
// value 110 allows SYCL users to run their destructors after runtime library
// deinitialization.
__attribute__((destructor(110))) static void syclUnload() { shutdown_late(); }
#endif
} // namespace detail
} // namespace _V1
} // namespace sycl

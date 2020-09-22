//==--------- global_handler.cpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/spinlock.hpp>
#include <detail/global_handler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
GlobalHandler *SyclGlobalObjectsHandler;
// SpinLock is chosen because, unlike std::mutex, it can be zero initialized,
// which spares us from dealing with global constructor/destructor call order.
SpinLock GlobalWritesAllowed;
SpinLock ShutdownLock;

GlobalHandler &GlobalHandler::instance() {
  if (!SyclGlobalObjectsHandler) {
    const std::lock_guard<SpinLock> Lock{GlobalWritesAllowed};
    if (!SyclGlobalObjectsHandler) {
      SyclGlobalObjectsHandler = new GlobalHandler();
    }
  }

  return *SyclGlobalObjectsHandler;
}

Scheduler &GlobalHandler::getScheduler() {
  if (!MScheduler) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MScheduler) {
      MScheduler = std::make_unique<Scheduler>();
    }
  }
  return *MScheduler;
}
ProgramManager &GlobalHandler::getProgramManager() {
  if (!MProgramManager) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MProgramManager) {
      MProgramManager = std::make_unique<ProgramManager>();
    }
  }
  return *MProgramManager;
}
Sync &GlobalHandler::getSync() {
  if (!MSync) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MSync) {
      MSync = std::make_unique<Sync>();
    }
  }
  return *MSync;
}
std::vector<PlatformImplPtr> &GlobalHandler::getPlatformCache() {
  if (!MPlatformCache) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MPlatformCache) {
      MPlatformCache = std::make_unique<std::vector<PlatformImplPtr>>();
    }
  }
  return *MPlatformCache;
}
std::mutex &GlobalHandler::getPlatformMapMutex() {
  if (!MPlatformMapMutex) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MPlatformMapMutex) {
      MPlatformMapMutex = std::make_unique<std::mutex>();
    }
  }
  return *MPlatformMapMutex;
}
std::mutex &GlobalHandler::getFilterMutex() {
  if (!MFilterMutex) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MFilterMutex) {
      MFilterMutex = std::make_unique<std::mutex>();
    }
  }
  return *MFilterMutex;
}
std::vector<plugin> &GlobalHandler::getPlugins() {
  if (!MPlugins) {
    const std::lock_guard<SpinLock> Lock{MFieldsLock};
    if (!MPlugins) {
      MPlugins = std::make_unique<std::vector<plugin>>();
    }
  }
  return *MPlugins;
}

void shutdown() {
  if (SyclGlobalObjectsHandler) {
    const std::lock_guard<SpinLock> Lock{ShutdownLock};
    if (SyclGlobalObjectsHandler) {
      // Acquire fields lock to make sure no thread creates fields while
      // object destruction is in flight.
      const std::lock_guard<SpinLock> Lock{
          SyclGlobalObjectsHandler->getFieldsLock()};
      delete SyclGlobalObjectsHandler;
    }
  }
}

#ifdef WIN32
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
__attribute__((destructor)) static void syclUnload() { shutdown(); }
#endif
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

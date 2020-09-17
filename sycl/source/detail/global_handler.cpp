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
// which spares us from dealing with constructor/destructor call order.
SpinLock GlobalWritesAllowed;

GlobalHandler &GlobalHandler::instance() {
  if (!SyclGlobalObjectsHandler) {
    const std::lock_guard<SpinLock> Lock{GlobalWritesAllowed};
    if (!SyclGlobalObjectsHandler) {
      SyclGlobalObjectsHandler = new GlobalHandler();
    }
  }

  return *SyclGlobalObjectsHandler;
}

static void shutdown() {
  if (SyclGlobalObjectsHandler) {
    const std::lock_guard<SpinLock> Lock{GlobalWritesAllowed};
    if (SyclGlobalObjectsHandler) {
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

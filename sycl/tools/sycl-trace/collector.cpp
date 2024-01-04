//==---------------------- collector.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.h"

#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>
#include <sycl/detail/spinlock.hpp>

sycl::detail::SpinLock GlobalLock;

bool HasZEPrinter = false;

std::string getCurrentDSODir() {
  auto CurrentFunc = reinterpret_cast<const void *>(&getCurrentDSODir);
  Dl_info Info;
  int RetCode = dladdr(CurrentFunc, &Info);
  if (0 == RetCode) {
    // This actually indicates an error
    return "";
  }

  auto Path = std::string(Info.dli_fname);
  auto LastSlashPos = Path.find_last_of('/');

  return Path.substr(0, LastSlashPos);
}

class CollectorLibraryWrapper {
  typedef void (*InitFuncType)();
  typedef void (*FinishFuncType)();
  typedef void (*CallbackFuncType)(uint16_t, xpti::trace_event_data_t *,
                                   xpti::trace_event_data_t *, uint64_t,
                                   const void *);
  typedef void (*SetIndentLvlFuncType)(int);

public:
  CollectorLibraryWrapper(const std::string &LibraryName)
      : MLibraryName(LibraryName){};
  ~CollectorLibraryWrapper() { clear(); };

  const std::string InitFuncName = "init";
  const std::string FinishFuncName = "finish";
  const std::string CallbackFuncName = "callback";
  const std::string IndentFuncName = "setIndentationLevel";

  bool initPrinters() {
    std::string Path = getCurrentDSODir();
    if (Path.empty())
      return false;
    Path += "/" + MLibraryName;
    MHandle = dlopen(Path.c_str(), RTLD_LAZY);
    if (!MHandle) {
      std::cerr << "Cannot load library: " << dlerror() << '\n';
      return false;
    }
    auto ExportSymbol = [&](void *&FuncPtr, const std::string &FuncName) {
      FuncPtr = dlsym(MHandle, FuncName.c_str());
      if (!FuncPtr) {
        std::cerr << "Cannot export symbol: " << dlerror() << '\n';
        return false;
      }
      return true;
    };
    if (!ExportSymbol(MInitPtr, InitFuncName) ||
        !ExportSymbol(MFinishPtr, FinishFuncName) ||
        !ExportSymbol(MSetIndentationLevelPtr, IndentFuncName) ||
        !ExportSymbol(MCallbackPtr, CallbackFuncName)) {
      clear();
      return false;
    }

    if (MIndentationLevel)
      ((SetIndentLvlFuncType)MSetIndentationLevelPtr)(MIndentationLevel);

    ((InitFuncType)MInitPtr)();

    return true;
  }

  void finishPrinters() {
    if (MHandle)
      ((FinishFuncType)MFinishPtr)();
  }

  void setIndentationLevel(int Level) {
    MIndentationLevel = Level;
    if (MHandle)
      ((SetIndentLvlFuncType)MSetIndentationLevelPtr)(MIndentationLevel);
  }

  void callback(uint16_t TraceType, xpti::trace_event_data_t *Parent,
                xpti::trace_event_data_t *Event, uint64_t Instance,
                const void *UserData) {
    // Not expected to be called when MHandle == NULL since we should not be
    // subscribed if init failed. Although still do the check for sure.
    if (MHandle)
      ((CallbackFuncType)MCallbackPtr)(TraceType, Parent, Event, Instance,
                                       UserData);
  }

  void clear() {
    MInitPtr = nullptr;
    MFinishPtr = nullptr;
    MCallbackPtr = nullptr;
    MSetIndentationLevelPtr = nullptr;

    if (MHandle)
      dlclose(MHandle);
    MHandle = nullptr;
  }

private:
  std::string MLibraryName;
  int MIndentationLevel = 0;

  void *MHandle = nullptr;

  void *MInitPtr = nullptr;
  void *MFinishPtr = nullptr;
  void *MCallbackPtr = nullptr;
  void *MSetIndentationLevelPtr = nullptr;
} zeCollectorLibrary("libze_trace_collector.so"),
    cudaCollectorLibrary("libcuda_trace_collector.so");

// These routing functions are needed to be able to use GlobalLock for
// dynamically loaded collectors.
XPTI_CALLBACK_API void zeCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData) {
  std::lock_guard<sycl::detail::SpinLock> _{GlobalLock};
  return zeCollectorLibrary.callback(TraceType, Parent, Event, Instance,
                                     UserData);
}
#ifdef USE_PI_CUDA
XPTI_CALLBACK_API void cudaCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *Parent,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t Instance, const void *UserData) {
  std::lock_guard<sycl::detail::SpinLock> _{GlobalLock};
  return cudaCollectorLibrary.callback(TraceType, Parent, Event, Instance,
                                       UserData);
}
#endif

void piPrintersInit();
void piPrintersFinish();
void syclPrintersInit();
void syclPrintersFinish();
void vPrintersInit();
void vPrintersFinish();

XPTI_CALLBACK_API void piCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *Parent,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void vCallback(uint16_t TraceType,
                                 xpti::trace_event_data_t *Parent,
                                 xpti::trace_event_data_t *Event,
                                 uint64_t Instance, const void *UserData);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE")) {
    piPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         piCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         piCallback);
    zeCollectorLibrary.setIndentationLevel(1);
    cudaCollectorLibrary.setIndentationLevel(1);
#ifdef SYCL_HAS_LEVEL_ZERO
  } else if (std::string_view(StreamName) ==
                 "sycl.experimental.level_zero.debug" &&
             std::getenv("SYCL_TRACE_ZE_ENABLE")) {
    if (zeCollectorLibrary.initPrinters()) {
      HasZEPrinter = true;
      uint16_t StreamID = xptiRegisterStream(StreamName);
      xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                           zeCallback);
      xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                           zeCallback);
    }
#endif
#ifdef USE_PI_CUDA
  } else if (std::string_view(StreamName) == "sycl.experimental.cuda.debug" &&
             std::getenv("SYCL_TRACE_CU_ENABLE")) {
    if (cudaCollectorLibrary.initPrinters()) {
      uint16_t StreamID = xptiRegisterStream(StreamName);
      xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                           cudaCallback);
      xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                           cudaCallback);
    }
#endif
  }
  if (std::string_view(StreamName) == "sycl" &&
      std::getenv("SYCL_TRACE_API_ENABLE")) {
    syclPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_diagnostics, syclCallback);
    xptiRegisterCallback(StreamID, xpti::trace_task_begin, syclCallback);
    xptiRegisterCallback(StreamID, xpti::trace_task_end, syclCallback);
    xptiRegisterCallback(StreamID, xpti::trace_queue_create, syclCallback);
    xptiRegisterCallback(StreamID, xpti::trace_queue_destroy, syclCallback);
  }
  if (std::getenv("SYCL_TRACE_VERIFICATION_ENABLE")) {
    vPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         vCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         vCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE"))
    piPrintersFinish();
#ifdef SYCL_HAS_LEVEL_ZERO
  else if (std::string_view(StreamName) ==
               "sycl.experimental.level_zero.debug" &&
           std::getenv("SYCL_TRACE_ZE_ENABLE")) {
    zeCollectorLibrary.finishPrinters();
    zeCollectorLibrary.clear();
  }
#endif
#ifdef USE_PI_CUDA
  else if (std::string_view(StreamName) == "sycl.experimental.cuda.debug" &&
           std::getenv("SYCL_TRACE_CU_ENABLE")) {
    cudaCollectorLibrary.finishPrinters();
    cudaCollectorLibrary.clear();
  }
#endif
  if (std::string_view(StreamName) == "sycl" &&
      std::getenv("SYCL_TRACE_API_ENABLE"))
    syclPrintersFinish();
  if (std::getenv("SYCL_TRACE_VERIFICATION_ENABLE")) {
    vPrintersFinish();
  }
}

diff --git a/sycl/test-e2e/AddressSanitizer/common/options-invalid-values.cpp b/sycl/test-e2e/AddressSanitizer/common/options-invalid-values.cpp
index ebe657d8f7642..7f57c3055e8a7 100644
--- a/sycl/test-e2e/AddressSanitizer/common/options-invalid-values.cpp
+++ b/sycl/test-e2e/AddressSanitizer/common/options-invalid-values.cpp
@@ -14,13 +14,13 @@
 // Invalid quarantine_size_mb
 // RUN: env UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:-1 %{run} not --crash %t 2>&1 | FileCheck %s  --check-prefixes INVALID-QUARANTINE
 // RUN: env UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:4294967296 %{run} not --crash %t 2>&1 | FileCheck %s  --check-prefixes INVALID-QUARANTINE
-// INVALID-QUARANTINE: <SANITIZER>[ERROR]: "quarantine_size_mb" should be an integer in range[0, 4294967295].
+// INVALID-QUARANTINE: <SANITIZER>[WARNING]: The valid range of "quarantine_size_mb" is [0, 4294967295].
 
 // Invalid redzone and max_redzone
 // RUN: env UR_LAYER_ASAN_OPTIONS=redzone:abc %{run} not --crash %t 2>&1 | FileCheck %s  --check-prefixes INVALID-REDZONE
-// INVALID-REDZONE: <SANITIZER>[ERROR]: "redzone" should be an integer in range[0, 16].
+// INVALID-REDZONE: <SANITIZER>[WARNING]: The valid range of redzone is [16, 2048].
 // RUN: env UR_LAYER_ASAN_OPTIONS=max_redzone:abc %{run} not --crash %t 2>&1 | FileCheck %s  --check-prefixes INVALID-MAXREDZONE
-// INVALID-MAXREDZONE: <SANITIZER>[ERROR]: "max_redzone" should be an integer in range[0, 2048].
+// INVALID-MAXREDZONE: <SANITIZER>[WARNING]: The valid range of "max_redzone" is [16, 2048].
 // clang-format on
 
 #include <sycl/usm.hpp>
diff --git a/sycl/test-e2e/AddressSanitizer/common/options-redzone.cpp b/sycl/test-e2e/AddressSanitizer/common/options-redzone.cpp
index 8f96415587572..7d4f0adf748c5 100644
--- a/sycl/test-e2e/AddressSanitizer/common/options-redzone.cpp
+++ b/sycl/test-e2e/AddressSanitizer/common/options-redzone.cpp
@@ -25,8 +25,8 @@ int main() {
   // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
   // CHECK: {{READ of size 1 at kernel <.*Test> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
   // CHECK: {{  #0 .* .*options-redzone.cpp:}}[[@LINE-7]]
-  // CHECK-MIN: Trying to set redzone size to a value less than 16 is ignored
-  // CHECK-MAX: Trying to set max redzone size to a value greater than 2048 is ignored
+  // CHECK-MIN: The valid range of "redzone" is [16, 2048]. Setting to the minimum value 16.
+  // CHECK-MAX: The valid range of "max_redzone" is [16, 2048]. Setting to the maximum value 2048.
 
   sycl::free(array, q);
   return 0;
diff --git a/unified-runtime/source/loader/CMakeLists.txt b/unified-runtime/source/loader/CMakeLists.txt
index 931c9dd3edb18..c03ba236baf85 100644
--- a/unified-runtime/source/loader/CMakeLists.txt
+++ b/unified-runtime/source/loader/CMakeLists.txt
@@ -140,8 +140,6 @@ if(UR_ENABLE_SANITIZER)
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_interceptor.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_interceptor.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_libdevice.hpp
-        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_options.cpp
-        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_options.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_quarantine.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_quarantine.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan/asan_report.cpp
@@ -161,8 +159,6 @@ if(UR_ENABLE_SANITIZER)
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_interceptor.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_interceptor.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_libdevice.hpp
-        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_options.cpp
-        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_options.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_report.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_report.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/msan/msan_shadow.cpp
@@ -176,6 +172,8 @@ if(UR_ENABLE_SANITIZER)
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/sanitizer_common/sanitizer_stacktrace.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/sanitizer_common/sanitizer_utils.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/sanitizer_common/sanitizer_utils.hpp
+        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/sanitizer_common/sanitizer_options.cpp
+        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/sanitizer_common/sanitizer_options.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/ur_sanddi.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/ur_sanitizer_layer.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/ur_sanitizer_layer.hpp
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_ddi.cpp b/unified-runtime/source/loader/layers/sanitizer/asan/asan_ddi.cpp
index 7b0c2e581f007..d2bbb8ded5cd3 100644
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_ddi.cpp
+++ b/unified-runtime/source/loader/layers/sanitizer/asan/asan_ddi.cpp
@@ -13,7 +13,6 @@
 
 #include "asan_ddi.hpp"
 #include "asan_interceptor.hpp"
-#include "asan_options.hpp"
 #include "sanitizer_common/sanitizer_stacktrace.hpp"
 #include "sanitizer_common/sanitizer_utils.hpp"
 #include "ur_sanitizer_layer.hpp"
@@ -1559,7 +1558,7 @@ __urdlllocal ur_result_t UR_APICALL urKernelSetArgPointer(
       pArgValue);
 
   std::shared_ptr<KernelInfo> KI;
-  if (getAsanInterceptor()->getOptions().DetectKernelArguments) {
+  if (getContext()->Options.DetectKernelArguments) {
     auto &KI = getAsanInterceptor()->getOrCreateKernelInfo(hKernel);
     std::scoped_lock<ur_shared_mutex> Guard(KI.Mutex);
     KI.PointerArgs[argIndex] = {pArgValue, GetCurrentBacktrace()};
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.cpp b/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.cpp
index d99c0545c3951..9b1809beed9ab 100644
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.cpp
+++ b/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.cpp
@@ -14,11 +14,11 @@
 
 #include "asan_interceptor.hpp"
 #include "asan_ddi.hpp"
-#include "asan_options.hpp"
 #include "asan_quarantine.hpp"
 #include "asan_report.hpp"
 #include "asan_shadow.hpp"
 #include "asan_validator.hpp"
+#include "sanitizer_common/sanitizer_options.hpp"
 #include "sanitizer_common/sanitizer_stacktrace.hpp"
 #include "sanitizer_common/sanitizer_utils.hpp"
 
@@ -26,9 +26,9 @@ namespace ur_sanitizer_layer {
 namespace asan {
 
 AsanInterceptor::AsanInterceptor() {
-  if (getOptions().MaxQuarantineSizeMB) {
+  if (getContext()->Options.MaxQuarantineSizeMB) {
     m_Quarantine = std::make_unique<Quarantine>(
-        static_cast<uint64_t>(getOptions().MaxQuarantineSizeMB) * 1024 * 1024);
+        getContext()->Options.MaxQuarantineSizeMB * 1024 * 1024);
   }
 }
 
@@ -90,8 +90,8 @@ ur_result_t AsanInterceptor::allocateMemory(ur_context_handle_t Context,
     Alignment = MinAlignment;
   }
 
-  uptr RZLog =
-      ComputeRZLog(Size, getOptions().MinRZSize, getOptions().MaxRZSize);
+  uptr RZLog = ComputeRZLog(Size, getContext()->Options.MinRZSize,
+                            getContext()->Options.MaxRZSize);
   uptr RZSize = RZLog2Size(RZLog);
   uptr RoundedSize = RoundUpTo(Size, Alignment);
   uptr NeededSize = RoundedSize + RZSize * 2;
@@ -175,7 +175,7 @@ ur_result_t AsanInterceptor::releaseMemory(ur_context_handle_t Context,
   if (!AllocInfoItOp) {
     // "Addr" might be a host pointer
     ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
-    if (getOptions().HaltOnError) {
+    if (getContext()->Options.HaltOnError) {
       exitWithErrors();
     }
     return UR_RESULT_SUCCESS;
@@ -193,7 +193,7 @@ ur_result_t AsanInterceptor::releaseMemory(ur_context_handle_t Context,
       // "Addr" might be a host pointer
       ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
     }
-    if (getOptions().HaltOnError) {
+    if (getContext()->Options.HaltOnError) {
       exitWithErrors();
     }
     return UR_RESULT_SUCCESS;
@@ -201,7 +201,7 @@ ur_result_t AsanInterceptor::releaseMemory(ur_context_handle_t Context,
 
   if (Addr != AllocInfo->UserBegin) {
     ReportBadFree(Addr, GetCurrentBacktrace(), AllocInfo);
-    if (getOptions().HaltOnError) {
+    if (getContext()->Options.HaltOnError) {
       exitWithErrors();
     }
     return UR_RESULT_SUCCESS;
@@ -209,7 +209,7 @@ ur_result_t AsanInterceptor::releaseMemory(ur_context_handle_t Context,
 
   if (AllocInfo->IsReleased) {
     ReportDoubleFree(Addr, GetCurrentBacktrace(), AllocInfo);
-    if (getOptions().HaltOnError) {
+    if (getContext()->Options.HaltOnError) {
       exitWithErrors();
     }
     return UR_RESULT_SUCCESS;
@@ -736,7 +736,7 @@ ur_result_t AsanInterceptor::prepareLaunch(
       LocalMemoryUsage, PrivateMemoryUsage);
 
   // Validate pointer arguments
-  if (getOptions().DetectKernelArguments) {
+  if (getContext()->Options.DetectKernelArguments) {
     for (const auto &[ArgIndex, PtrPair] : KernelInfo.PointerArgs) {
       auto Ptr = PtrPair.first;
       if (Ptr == nullptr) {
@@ -813,10 +813,10 @@ ur_result_t AsanInterceptor::prepareLaunch(
   LaunchInfo.Data.Host.GlobalShadowOffset = DeviceInfo->Shadow->ShadowBegin;
   LaunchInfo.Data.Host.GlobalShadowOffsetEnd = DeviceInfo->Shadow->ShadowEnd;
   LaunchInfo.Data.Host.DeviceTy = DeviceInfo->Type;
-  LaunchInfo.Data.Host.Debug = getOptions().Debug ? 1 : 0;
+  LaunchInfo.Data.Host.Debug = getContext()->Options.Debug ? 1 : 0;
 
   // Write shadow memory offset for local memory
-  if (getOptions().DetectLocals) {
+  if (getContext()->Options.DetectLocals) {
     if (DeviceInfo->Shadow->AllocLocalShadow(
             Queue, NumWG, LaunchInfo.Data.Host.LocalShadowOffset,
             LaunchInfo.Data.Host.LocalShadowOffsetEnd) != UR_RESULT_SUCCESS) {
@@ -836,7 +836,7 @@ ur_result_t AsanInterceptor::prepareLaunch(
   }
 
   // Write shadow memory offset for private memory
-  if (getOptions().DetectPrivates) {
+  if (getContext()->Options.DetectPrivates) {
     if (DeviceInfo->Shadow->AllocPrivateShadow(
             Queue, NumWG, LaunchInfo.Data.Host.PrivateShadowOffset,
             LaunchInfo.Data.Host.PrivateShadowOffsetEnd) != UR_RESULT_SUCCESS) {
@@ -928,7 +928,7 @@ ContextInfo::~ContextInfo() {
   assert(URes == UR_RESULT_SUCCESS);
 
   // check memory leaks
-  if (getAsanInterceptor()->getOptions().DetectLeaks &&
+  if (getContext()->Options.DetectLeaks &&
       getAsanInterceptor()->isNormalExit()) {
     std::vector<AllocationIterator> AllocInfos =
         getAsanInterceptor()->findAllocInfoByContext(Handle);
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.hpp b/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.hpp
index 3f0a4642c9983..53aac416f2567 100644
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.hpp
+++ b/unified-runtime/source/loader/layers/sanitizer/asan/asan_interceptor.hpp
@@ -16,10 +16,10 @@
 #include "asan_allocator.hpp"
 #include "asan_buffer.hpp"
 #include "asan_libdevice.hpp"
-#include "asan_options.hpp"
 #include "asan_shadow.hpp"
 #include "asan_statistics.hpp"
 #include "sanitizer_common/sanitizer_common.hpp"
+#include "sanitizer_common/sanitizer_options.hpp"
 #include "ur_sanitizer_layer.hpp"
 
 #include <memory>
@@ -342,8 +342,6 @@ class AsanInterceptor {
   KernelInfo &getOrCreateKernelInfo(ur_kernel_handle_t Kernel);
   ur_result_t eraseKernelInfo(ur_kernel_handle_t Kernel);
 
-  const AsanOptions &getOptions() { return m_Options; }
-
   void exitWithErrors() {
     m_NormalExit = false;
     exit(1);
@@ -375,8 +373,6 @@ class AsanInterceptor {
   ur_result_t registerSpirKernels(ur_program_handle_t Program);
 
 private:
-  // m_Options may be used in other places, place it at the top
-  AsanOptions m_Options;
   std::unordered_map<ur_context_handle_t, std::shared_ptr<ContextInfo>>
       m_ContextMap;
   ur_shared_mutex m_ContextMapMutex;
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_options.cpp b/unified-runtime/source/loader/layers/sanitizer/asan/asan_options.cpp
deleted file mode 100644
index 799c892ff153d..0000000000000
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_options.cpp
+++ /dev/null
@@ -1,148 +0,0 @@
-/*
- *
- * Copyright (C) 2024 Intel Corporation
- *
- * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
- * Exceptions. See LICENSE.TXT
- *
- * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
- *
- * @file asan_options.cpp
- *
- */
-
-#include "asan_options.hpp"
-
-#include "ur/ur.hpp"
-#include "ur_sanitizer_layer.hpp"
-
-#include <algorithm>
-#include <cstring>
-#include <stdexcept>
-
-namespace ur_sanitizer_layer {
-namespace asan {
-
-AsanOptions::AsanOptions() {
-  std::optional<EnvVarMap> OptionsEnvMap;
-  try {
-    OptionsEnvMap = getenv_to_map("UR_LAYER_ASAN_OPTIONS");
-  } catch (const std::invalid_argument &e) {
-    std::stringstream SS;
-    SS << "<SANITIZER>[ERROR]: ";
-    SS << e.what();
-    getContext()->logger.always(SS.str().c_str());
-    die("Sanitizer failed to parse options.\n");
-  }
-
-  if (!OptionsEnvMap.has_value()) {
-    return;
-  }
-
-  const char *TrueStrings[] = {"1", "true"};
-  const char *FalseStrings[] = {"0", "false"};
-
-  auto InplaceToLower = [](std::string &S) {
-    std::transform(S.begin(), S.end(), S.begin(),
-                   [](unsigned char C) { return std::tolower(C); });
-  };
-  auto IsTrue = [&](const std::string &S) {
-    return std::any_of(std::begin(TrueStrings), std::end(TrueStrings),
-                       [&](const char *CS) { return S == CS; });
-  };
-  auto IsFalse = [&](const std::string &S) {
-    return std::any_of(std::begin(FalseStrings), std::end(FalseStrings),
-                       [&](const char *CS) { return S == CS; });
-  };
-
-  auto SetBoolOption = [&](const std::string &Name, bool &Opt) {
-    auto KV = OptionsEnvMap->find(Name);
-    if (KV != OptionsEnvMap->end()) {
-      auto Value = KV->second.front();
-      InplaceToLower(Value);
-      if (IsTrue(Value)) {
-        Opt = true;
-      } else if (IsFalse(Value)) {
-        Opt = false;
-      } else {
-        std::stringstream SS;
-        SS << "\"" << Name << "\" is set to \"" << Value
-           << "\", which is not an valid setting. ";
-        SS << "Acceptable input are: for enable, use:";
-        for (auto &S : TrueStrings) {
-          SS << " \"" << S << "\"";
-        }
-        SS << "; ";
-        SS << "for disable, use:";
-        for (auto &S : FalseStrings) {
-          SS << " \"" << S << "\"";
-        }
-        SS << ".";
-        getContext()->logger.error(SS.str().c_str());
-        die("Sanitizer failed to parse options.\n");
-      }
-    }
-  };
-
-  SetBoolOption("debug", Debug);
-  SetBoolOption("detect_kernel_arguments", DetectKernelArguments);
-  SetBoolOption("detect_locals", DetectLocals);
-  SetBoolOption("detect_privates", DetectPrivates);
-  SetBoolOption("print_stats", PrintStats);
-  SetBoolOption("detect_leaks", DetectLeaks);
-  SetBoolOption("halt_on_error", HaltOnError);
-
-  auto KV = OptionsEnvMap->find("quarantine_size_mb");
-  if (KV != OptionsEnvMap->end()) {
-    const auto &Value = KV->second.front();
-    try {
-      auto temp_long = std::stoul(Value);
-      if (temp_long > UINT32_MAX) {
-        throw std::out_of_range("");
-      }
-      MaxQuarantineSizeMB = temp_long;
-    } catch (...) {
-      getContext()->logger.error("\"quarantine_size_mb\" should be "
-                                 "an integer in range[0, {}].",
-                                 UINT32_MAX);
-      die("Sanitizer failed to parse options.\n");
-    }
-  }
-
-  KV = OptionsEnvMap->find("redzone");
-  if (KV != OptionsEnvMap->end()) {
-    const auto &Value = KV->second.front();
-    try {
-      MinRZSize = std::stoul(Value);
-      if (MinRZSize < 16) {
-        MinRZSize = 16;
-        getContext()->logger.warning("Trying to set redzone size to a "
-                                     "value less than 16 is ignored.");
-      }
-    } catch (...) {
-      getContext()->logger.error(
-          "\"redzone\" should be an integer in range[0, 16].");
-      die("Sanitizer failed to parse options.\n");
-    }
-  }
-
-  KV = OptionsEnvMap->find("max_redzone");
-  if (KV != OptionsEnvMap->end()) {
-    const auto &Value = KV->second.front();
-    try {
-      MaxRZSize = std::stoul(Value);
-      if (MaxRZSize > 2048) {
-        MaxRZSize = 2048;
-        getContext()->logger.warning("Trying to set max redzone size to a "
-                                     "value greater than 2048 is ignored.");
-      }
-    } catch (...) {
-      getContext()->logger.error(
-          "\"max_redzone\" should be an integer in range[0, 2048].");
-      die("Sanitizer failed to parse options.\n");
-    }
-  }
-}
-
-} // namespace asan
-} // namespace ur_sanitizer_layer
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_report.cpp b/unified-runtime/source/loader/layers/sanitizer/asan/asan_report.cpp
index a318f03e70020..2f294f77faf24 100644
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_report.cpp
+++ b/unified-runtime/source/loader/layers/sanitizer/asan/asan_report.cpp
@@ -15,8 +15,8 @@
 #include "asan_allocator.hpp"
 #include "asan_interceptor.hpp"
 #include "asan_libdevice.hpp"
-#include "asan_options.hpp"
 #include "asan_validator.hpp"
+#include "sanitizer_common/sanitizer_options.hpp"
 #include "sanitizer_common/sanitizer_utils.hpp"
 #include "ur_sanitizer_layer.hpp"
 
@@ -135,7 +135,7 @@ void ReportUseAfterFree(const AsanErrorReport &Report,
   getContext()->logger.always("  #0 {} {}:{}", Func, File, Report.Line);
   getContext()->logger.always("");
 
-  if (getAsanInterceptor()->getOptions().MaxQuarantineSizeMB > 0) {
+  if (getContext()->Options.MaxQuarantineSizeMB > 0) {
     auto AllocInfoItOp =
         getAsanInterceptor()->findAllocInfoByAddress(Report.Address);
 
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_statistics.cpp b/unified-runtime/source/loader/layers/sanitizer/asan/asan_statistics.cpp
index a639bf7cfe12d..a149245f345fe 100644
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_statistics.cpp
+++ b/unified-runtime/source/loader/layers/sanitizer/asan/asan_statistics.cpp
@@ -68,7 +68,7 @@ void AsanStats::UpdateUSMFreed(uptr FreedSize) {
 void AsanStats::UpdateUSMRealFreed(uptr FreedSize, uptr RedzoneSize) {
   UsmMalloced -= FreedSize;
   UsmMallocedRedzones -= RedzoneSize;
-  if (getAsanInterceptor()->getOptions().MaxQuarantineSizeMB) {
+  if (getContext()->Options.MaxQuarantineSizeMB) {
     UsmFreed -= FreedSize;
   }
   getContext()->logger.debug(
@@ -137,7 +137,7 @@ void AsanStatsWrapper::Print(ur_context_handle_t Context) {
 }
 
 AsanStatsWrapper::AsanStatsWrapper() : Stat(nullptr) {
-  if (getAsanInterceptor()->getOptions().PrintStats) {
+  if (getContext()->Options.PrintStats) {
     Stat = new AsanStats;
   }
 }
diff --git a/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.cpp b/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.cpp
index 7994f1fce446b..9da8dad0b91ab 100644
--- a/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.cpp
+++ b/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.cpp
@@ -14,7 +14,6 @@
 
 #include "msan_interceptor.hpp"
 #include "msan_ddi.hpp"
-#include "msan_options.hpp"
 #include "msan_report.hpp"
 #include "msan_shadow.hpp"
 #include "sanitizer_common/sanitizer_stacktrace.hpp"
@@ -453,7 +452,7 @@ ur_result_t MsanInterceptor::prepareLaunch(
   LaunchInfo.Data->GlobalShadowOffset = DeviceInfo->Shadow->ShadowBegin;
   LaunchInfo.Data->GlobalShadowOffsetEnd = DeviceInfo->Shadow->ShadowEnd;
   LaunchInfo.Data->DeviceTy = DeviceInfo->Type;
-  LaunchInfo.Data->Debug = getOptions().Debug ? 1 : 0;
+  LaunchInfo.Data->Debug = getContext()->Options.Debug ? 1 : 0;
   UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
       ContextInfo->Handle, DeviceInfo->Handle, nullptr, nullptr,
       ContextInfo->MaxAllocatedSize, &LaunchInfo.Data->CleanShadow));
diff --git a/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.hpp b/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.hpp
index a7e1d96d7edbe..dc238d45337f4 100644
--- a/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.hpp
+++ b/unified-runtime/source/loader/layers/sanitizer/msan/msan_interceptor.hpp
@@ -16,9 +16,9 @@
 #include "msan_allocator.hpp"
 #include "msan_buffer.hpp"
 #include "msan_libdevice.hpp"
-#include "msan_options.hpp"
 #include "msan_shadow.hpp"
 #include "sanitizer_common/sanitizer_common.hpp"
+#include "sanitizer_common/sanitizer_options.hpp"
 #include "ur_sanitizer_layer.hpp"
 
 #include <memory>
@@ -247,8 +247,6 @@ class MsanInterceptor {
   KernelInfo &getOrCreateKernelInfo(ur_kernel_handle_t Kernel);
   ur_result_t eraseKernelInfo(ur_kernel_handle_t Kernel);
 
-  const MsanOptions &getOptions() { return m_Options; }
-
   void exitWithErrors() {
     m_NormalExit = false;
     exit(1);
@@ -292,8 +290,6 @@ class MsanInterceptor {
   MsanAllocationMap m_AllocationMap;
   ur_shared_mutex m_AllocationMapMutex;
 
-  MsanOptions m_Options;
-
   std::unordered_set<ur_adapter_handle_t> m_Adapters;
   ur_shared_mutex m_AdaptersMutex;
 
diff --git a/unified-runtime/source/loader/layers/sanitizer/msan/msan_options.cpp b/unified-runtime/source/loader/layers/sanitizer/msan/msan_options.cpp
deleted file mode 100644
index a93ac65b1f53d..0000000000000
--- a/unified-runtime/source/loader/layers/sanitizer/msan/msan_options.cpp
+++ /dev/null
@@ -1,91 +0,0 @@
-/*
- *
- * Copyright (C) 2024 Intel Corporation
- *
- * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
- * Exceptions. See LICENSE.TXT
- *
- * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
- *
- * @file msan_options.cpp
- *
- */
-
-#include "msan_options.hpp"
-
-#include "ur/ur.hpp"
-#include "ur_sanitizer_layer.hpp"
-
-#include <algorithm>
-#include <cstring>
-#include <stdexcept>
-
-namespace ur_sanitizer_layer {
-namespace msan {
-
-MsanOptions::MsanOptions() {
-  std::optional<EnvVarMap> OptionsEnvMap;
-  try {
-    OptionsEnvMap = getenv_to_map("UR_LAYER_MSAN_OPTIONS");
-  } catch (const std::invalid_argument &e) {
-    std::stringstream SS;
-    SS << "<SANITIZER>[ERROR]: ";
-    SS << e.what();
-    getContext()->logger.always(SS.str().c_str());
-    die("Sanitizer failed to parse options.\n");
-  }
-
-  if (!OptionsEnvMap.has_value()) {
-    return;
-  }
-
-  const char *TrueStrings[] = {"1", "true"};
-  const char *FalseStrings[] = {"0", "false"};
-
-  auto InplaceToLower = [](std::string &S) {
-    std::transform(S.begin(), S.end(), S.begin(),
-                   [](unsigned char C) { return std::tolower(C); });
-  };
-  auto IsTrue = [&](const std::string &S) {
-    return std::any_of(std::begin(TrueStrings), std::end(TrueStrings),
-                       [&](const char *CS) { return S == CS; });
-  };
-  auto IsFalse = [&](const std::string &S) {
-    return std::any_of(std::begin(FalseStrings), std::end(FalseStrings),
-                       [&](const char *CS) { return S == CS; });
-  };
-
-  auto SetBoolOption = [&](const std::string &Name, bool &Opt) {
-    auto KV = OptionsEnvMap->find(Name);
-    if (KV != OptionsEnvMap->end()) {
-      auto Value = KV->second.front();
-      InplaceToLower(Value);
-      if (IsTrue(Value)) {
-        Opt = true;
-      } else if (IsFalse(Value)) {
-        Opt = false;
-      } else {
-        std::stringstream SS;
-        SS << "\"" << Name << "\" is set to \"" << Value
-           << "\", which is not an valid setting. ";
-        SS << "Acceptable input are: for enable, use:";
-        for (auto &S : TrueStrings) {
-          SS << " \"" << S << "\"";
-        }
-        SS << "; ";
-        SS << "for disable, use:";
-        for (auto &S : FalseStrings) {
-          SS << " \"" << S << "\"";
-        }
-        SS << ".";
-        getContext()->logger.error(SS.str().c_str());
-        die("Sanitizer failed to parse options.\n");
-      }
-    }
-  };
-
-  SetBoolOption("debug", Debug);
-}
-
-} // namespace msan
-} // namespace ur_sanitizer_layer
diff --git a/unified-runtime/source/loader/layers/sanitizer/msan/msan_options.hpp b/unified-runtime/source/loader/layers/sanitizer/msan/msan_options.hpp
deleted file mode 100644
index bc24d0427e9ed..0000000000000
--- a/unified-runtime/source/loader/layers/sanitizer/msan/msan_options.hpp
+++ /dev/null
@@ -1,28 +0,0 @@
-/*
- *
- * Copyright (C) 2024 Intel Corporation
- *
- * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
- * Exceptions. See LICENSE.TXT
- *
- * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
- *
- * @file msan_options.hpp
- *
- */
-
-#pragma once
-
-#include <cstdint>
-
-namespace ur_sanitizer_layer {
-namespace msan {
-
-struct MsanOptions {
-  bool Debug = false;
-
-  explicit MsanOptions();
-};
-
-} // namespace msan
-} // namespace ur_sanitizer_layer
diff --git a/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.cpp b/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.cpp
new file mode 100644
index 0000000000000..20460be26f15a
--- /dev/null
+++ b/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.cpp
@@ -0,0 +1,54 @@
+/*
+ *
+ * Copyright (C) 2025 Intel Corporation
+ *
+ * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
+ * Exceptions. See LICENSE.TXT
+ *
+ * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+ *
+ * @file sanitizer_options.cpp
+ *
+ */
+
+#include "sanitizer_options.hpp"
+#include "sanitizer_options_impl.hpp"
+
+#include <cstring>
+#include <stdexcept>
+
+namespace ur_sanitizer_layer {
+
+void SanitizerOptions::Init(const std::string &EnvName,
+                            logger::Logger &Logger) {
+  std::optional<EnvVarMap> OptionsEnvMap;
+  try {
+    OptionsEnvMap = getenv_to_map(EnvName.c_str());
+  } catch (const std::invalid_argument &e) {
+    std::stringstream SS;
+    SS << "<SANITIZER>[ERROR]: ";
+    SS << e.what();
+    Logger.always(SS.str().c_str());
+    die("Sanitizer failed to parse options.\n");
+  }
+
+  if (!OptionsEnvMap.has_value()) {
+    return;
+  }
+
+  auto Parser = options::OptionParser(OptionsEnvMap.value(), Logger);
+
+  Parser.ParseBool("debug", Debug);
+  Parser.ParseBool("detect_kernel_arguments", DetectKernelArguments);
+  Parser.ParseBool("detect_locals", DetectLocals);
+  Parser.ParseBool("detect_privates", DetectPrivates);
+  Parser.ParseBool("print_stats", PrintStats);
+  Parser.ParseBool("detect_leaks", DetectLeaks);
+  Parser.ParseBool("halt_on_error", HaltOnError);
+
+  Parser.ParseUint64("quarantine_size_mb", MaxQuarantineSizeMB, 0, UINT32_MAX);
+  Parser.ParseUint64("redzone", MinRZSize, 16, 2048);
+  Parser.ParseUint64("max_redzone", MaxRZSize, 16, 2048);
+}
+
+} // namespace ur_sanitizer_layer
diff --git a/unified-runtime/source/loader/layers/sanitizer/asan/asan_options.hpp b/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.hpp
similarity index 67%
rename from unified-runtime/source/loader/layers/sanitizer/asan/asan_options.hpp
rename to unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.hpp
index cea30351d3d49..cfdd5868448c9 100644
--- a/unified-runtime/source/loader/layers/sanitizer/asan/asan_options.hpp
+++ b/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.hpp
@@ -1,28 +1,31 @@
 /*
  *
- * Copyright (C) 2024 Intel Corporation
+ * Copyright (C) 2025 Intel Corporation
  *
  * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
  * Exceptions. See LICENSE.TXT
  *
  * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  *
- * @file asan_options.hpp
+ * @file sanitizer_options.hpp
  *
  */
 
 #pragma once
 
+#include "logger/ur_logger.hpp"
+#include "ur/ur.hpp"
+
 #include <cstdint>
+#include <string>
 
 namespace ur_sanitizer_layer {
-namespace asan {
 
-struct AsanOptions {
+struct SanitizerOptions {
   bool Debug = false;
   uint64_t MinRZSize = 16;
   uint64_t MaxRZSize = 2048;
-  uint32_t MaxQuarantineSizeMB = 8;
+  uint64_t MaxQuarantineSizeMB = 8;
   bool DetectLocals = true;
   bool DetectPrivates = true;
   bool PrintStats = false;
@@ -30,8 +33,7 @@ struct AsanOptions {
   bool DetectLeaks = true;
   bool HaltOnError = true;
 
-  explicit AsanOptions();
+  void Init(const std::string &EnvName, logger::Logger &Logger);
 };
 
-} // namespace asan
 } // namespace ur_sanitizer_layer
diff --git a/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options_impl.hpp b/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options_impl.hpp
new file mode 100644
index 0000000000000..fa77ef387d9c2
--- /dev/null
+++ b/unified-runtime/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options_impl.hpp
@@ -0,0 +1,111 @@
+/*
+ *
+ * Copyright (C) 2025 Intel Corporation
+ *
+ * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
+ * Exceptions. See LICENSE.TXT
+ *
+ * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+ *
+ * @file sanitizer_options_impl.hpp
+ *
+ */
+
+#pragma once
+
+#include "ur/ur.hpp"
+#include "ur_sanitizer_layer.hpp"
+
+#include <algorithm>
+#include <cstdint>
+#include <string>
+
+namespace ur_sanitizer_layer {
+
+namespace options {
+
+struct OptionParser {
+  logger::Logger &Logger;
+  const EnvVarMap &EnvMap;
+
+  OptionParser(const EnvVarMap &EnvMap, logger::Logger &Logger)
+      : Logger(Logger), EnvMap(EnvMap) {}
+
+  const char *TrueStrings[2] = {"1", "true"};
+  const char *FalseStrings[2] = {"0", "false"};
+
+  void InplaceToLower(std::string &S) {
+    std::transform(S.begin(), S.end(), S.begin(),
+                   [](unsigned char C) { return std::tolower(C); });
+  }
+
+  bool IsTrue(const std::string &S) {
+    return std::any_of(std::begin(TrueStrings), std::end(TrueStrings),
+                       [&](const char *CS) { return S == CS; });
+  }
+
+  bool IsFalse(const std::string &S) {
+    return std::any_of(std::begin(FalseStrings), std::end(FalseStrings),
+                       [&](const char *CS) { return S == CS; });
+  }
+  void ParseBool(const std::string &Name, bool &Result) {
+    auto KV = EnvMap.find(Name);
+    if (KV != EnvMap.end()) {
+      auto ValueStr = KV->second.front();
+      InplaceToLower(ValueStr);
+      if (IsTrue(ValueStr)) {
+        Result = true;
+      } else if (IsFalse(ValueStr)) {
+        Result = false;
+      } else {
+        std::stringstream SS;
+        SS << "\"" << Name << "\" is set to \"" << ValueStr
+           << "\", which is not an valid setting. ";
+        SS << "Acceptable input are: for enable, use:";
+        for (auto &S : TrueStrings) {
+          SS << " \"" << S << "\"";
+        }
+        SS << "; ";
+        SS << "for disable, use:";
+        for (auto &S : FalseStrings) {
+          SS << " \"" << S << "\"";
+        }
+        SS << ".";
+        Logger.error(SS.str().c_str());
+        die("Sanitizer failed to parse options.\n");
+      }
+    }
+  }
+
+  void ParseUint64(const std::string &Name, uint64_t &Result, uint64_t Min = 0,
+                   uint64_t Max = UINT64_MAX) {
+    auto KV = EnvMap.find(Name);
+    if (KV != EnvMap.end()) {
+      const auto &ValueStr = KV->second.front();
+      try {
+        uint64_t Value = std::stoul(ValueStr.c_str());
+        if (Value < Min) {
+          Logger.warning("The valid range of \"{}\" is [{}, {}]. "
+                         "Setting to the minimum value {}.",
+                         Name, Min, Max, Min);
+          Result = Min;
+        } else if (Value > Max) {
+          Logger.warning("The valid range of \"{}\" is [{}, {}]. "
+                         "Setting to the maximum value {}.",
+                         Name, Min, Max, Max);
+          Result = Max;
+        } else {
+          Result = Value;
+        }
+      } catch (...) {
+        Logger.error("The valid range of \"{}\" is [{}, {}]. Failed "
+                     "to parse the value \"{}\".",
+                     Name, Min, Max, ValueStr);
+        die("Sanitizer failed to parse options.\n");
+      }
+    }
+  }
+};
+} // namespace options
+
+} // namespace ur_sanitizer_layer
diff --git a/unified-runtime/source/loader/layers/sanitizer/ur_sanddi.cpp b/unified-runtime/source/loader/layers/sanitizer/ur_sanddi.cpp
index 4b1683020cbab..a32147946e273 100644
--- a/unified-runtime/source/loader/layers/sanitizer/ur_sanddi.cpp
+++ b/unified-runtime/source/loader/layers/sanitizer/ur_sanddi.cpp
@@ -40,9 +40,11 @@ ur_result_t context_t::init(ur_dditable_t *dditable,
 
   switch (enabledType) {
   case SanitizerType::AddressSanitizer:
+    getContext()->Options.Init("UR_LAYER_ASAN_OPTIONS", getContext()->logger);
     initAsanInterceptor();
     return initAsanDDITable(dditable);
   case SanitizerType::MemorySanitizer:
+    getContext()->Options.Init("UR_LAYER_MSAN_OPTIONS", getContext()->logger);
     initMsanInterceptor();
     return initMsanDDITable(dditable);
   default:
diff --git a/unified-runtime/source/loader/layers/sanitizer/ur_sanitizer_layer.hpp b/unified-runtime/source/loader/layers/sanitizer/ur_sanitizer_layer.hpp
index cd51f33c8a43e..1932e19cb2264 100644
--- a/unified-runtime/source/loader/layers/sanitizer/ur_sanitizer_layer.hpp
+++ b/unified-runtime/source/loader/layers/sanitizer/ur_sanitizer_layer.hpp
@@ -14,6 +14,7 @@
 #pragma once
 
 #include "logger/ur_logger.hpp"
+#include "sanitizer_common/sanitizer_options.hpp"
 #include "ur_proxy_layer.hpp"
 
 #define SANITIZER_COMP_NAME "sanitizer layer"
@@ -34,6 +35,7 @@ class __urdlllocal context_t : public proxy_layer_context_t,
   ur_dditable_t urDdiTable = {};
   logger::Logger logger;
   SanitizerType enabledType = SanitizerType::None;
+  SanitizerOptions Options;
 
   context_t();
   ~context_t();
diff --git a/unified-runtime/test/layers/sanitizer/CMakeLists.txt b/unified-runtime/test/layers/sanitizer/CMakeLists.txt
index a9601a89c886f..23c4b570af466 100644
--- a/unified-runtime/test/layers/sanitizer/CMakeLists.txt
+++ b/unified-runtime/test/layers/sanitizer/CMakeLists.txt
@@ -22,6 +22,9 @@ function(set_sanitizer_test_properties name)
     set_tests_properties(${name} PROPERTIES LABELS "sanitizer")
     set_property(TEST ${name} PROPERTY ENVIRONMENT
         "UR_LOG_SANITIZER=level:debug\;flush:debug\;output:stdout")
+    target_include_directories(${name} PRIVATE
+        ${PROJECT_SOURCE_DIR}/source/loader/layers/sanitizer/sanitizer_common
+    )
 endfunction()
 
 function(add_sanitizer_test name)
diff --git a/unified-runtime/test/layers/sanitizer/sanitizer_options.cpp b/unified-runtime/test/layers/sanitizer/sanitizer_options.cpp
new file mode 100644
index 0000000000000..7b3fba11dc2f4
--- /dev/null
+++ b/unified-runtime/test/layers/sanitizer/sanitizer_options.cpp
@@ -0,0 +1,123 @@
+/*
+ *
+ * Copyright (C) 2025 Intel Corporation
+ *
+ * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
+ * Exceptions. See LICENSE.TXT
+ *
+ * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+ *
+ * @file sanitizer_options.cpp
+ *
+ */
+
+#include "sanitizer_options.hpp"
+#include "sanitizer_options_impl.hpp"
+#include <gtest/gtest.h>
+
+TEST(DeviceAsan, Initialization) {
+  ur_result_t status;
+
+  ur_loader_config_handle_t loaderConfig;
+  status = urLoaderConfigCreate(&loaderConfig);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+  status = urLoaderConfigEnableLayer(loaderConfig, "UR_LAYER_ASAN");
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urLoaderInit(0, loaderConfig);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_adapter_handle_t adapter;
+  status = urAdapterGet(1, &adapter, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_platform_handle_t platform;
+  status = urPlatformGet(&adapter, 1, 1, &platform, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_device_handle_t device;
+  status = urDeviceGet(platform, UR_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_context_handle_t context;
+  status = urContextCreate(1, &device, nullptr, &context);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urContextRelease(context);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urDeviceRelease(device);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urAdapterRelease(adapter);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urLoaderTearDown();
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urLoaderConfigRelease(loaderConfig);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+}
+
+TEST(DeviceAsan, UnsupportedFeature) {
+  ur_result_t status;
+
+  ur_loader_config_handle_t loaderConfig;
+  status = urLoaderConfigCreate(&loaderConfig);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+  status = urLoaderConfigEnableLayer(loaderConfig, "UR_LAYER_ASAN");
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urLoaderInit(0, loaderConfig);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_adapter_handle_t adapter;
+  status = urAdapterGet(1, &adapter, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_platform_handle_t platform;
+  status = urPlatformGet(&adapter, 1, 1, &platform, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_device_handle_t device;
+  status = urDeviceGet(platform, UR_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  ur_context_handle_t context;
+  status = urContextCreate(1, &device, nullptr, &context);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  // Check for explict unsupported features
+  ur_bool_t isSupported;
+  status = urDeviceGetInfo(device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT,
+                           sizeof(isSupported), &isSupported, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+  ASSERT_EQ(isSupported, 0);
+
+  status = urDeviceGetInfo(device, UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP,
+                           sizeof(isSupported), &isSupported, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+  ASSERT_EQ(isSupported, 0);
+
+  ur_device_command_buffer_update_capability_flags_t update_flag;
+  status = urDeviceGetInfo(
+      device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
+      sizeof(update_flag), &update_flag, nullptr);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+  ASSERT_EQ(update_flag, 0);
+
+  status = urContextRelease(context);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urDeviceRelease(device);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urAdapterRelease(adapter);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urLoaderTearDown();
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+
+  status = urLoaderConfigRelease(loaderConfig);
+  ASSERT_EQ(status, UR_RESULT_SUCCESS);
+}

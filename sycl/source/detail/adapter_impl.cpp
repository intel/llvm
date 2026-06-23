//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definitions for the members of the adapter_impl
/// class.
///
//===----------------------------------------------------------------------===//

#include "adapter_impl.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>

#ifdef _WIN32
#include <windows.h>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

// DIAGNOSTIC (intel/llvm#22367). Teardown-safe append: ~Managed can run during
// late process teardown where C++ iostream is unreliable, so use the OS append
// primitive. No-op unless SYCL_TRACE_ADAPTER_UAF is set.
static void diagAppend(const char *Text, size_t Len) {
  const char *Path = std::getenv("SYCL_TRACE_ADAPTER_UAF");
  if (!Path || !*Path)
    return;
#ifdef _WIN32
  HANDLE H =
      CreateFileA(Path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                  nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (H == INVALID_HANDLE_VALUE)
    return;
  DWORD Written = 0;
  WriteFile(H, Text, static_cast<DWORD>(Len), &Written, nullptr);
  CloseHandle(H);
#else
  if (FILE *F = std::fopen(Path, "a")) {
    std::fwrite(Text, 1, Len, F);
    std::fclose(F);
  }
#endif
}

// Live adapter_impl pointers. Used to decide whether a Managed's adapter pointer
// is safe to dereference (still constructed) or stale (already destroyed).
static std::mutex &diagMutex() {
  static std::mutex *M = new std::mutex(); // leaked: must outlive all ~Managed
  return *M;
}
static std::set<const void *> &diagLiveAdapters() {
  static std::set<const void *> *S =
      new std::set<const void *>(); // leaked on purpose
  return *S;
}

void diagAdapterRegister(const void *Adapter) {
  if (!std::getenv("SYCL_TRACE_ADAPTER_UAF"))
    return;
  {
    std::lock_guard<std::mutex> Lock(diagMutex());
    diagLiveAdapters().insert(Adapter);
  }
  char Buf[64];
  int N = std::snprintf(Buf, sizeof(Buf), "ADAPTER_CTOR %p\n", Adapter);
  if (N > 0)
    diagAppend(Buf, static_cast<size_t>(N));
}

void diagAdapterUnregister(const void *Adapter) {
  if (!std::getenv("SYCL_TRACE_ADAPTER_UAF"))
    return;
  {
    std::lock_guard<std::mutex> Lock(diagMutex());
    diagLiveAdapters().erase(Adapter);
  }
  char Buf[64];
  int N = std::snprintf(Buf, sizeof(Buf), "ADAPTER_DTOR %p\n", Adapter);
  if (N > 0)
    diagAppend(Buf, static_cast<size_t>(N));
}

void diagManagedRelease(const void *Adapter, const void *Resource) {
  if (!std::getenv("SYCL_TRACE_ADAPTER_UAF"))
    return;
  bool Live;
  {
    std::lock_guard<std::mutex> Lock(diagMutex());
    Live = diagLiveAdapters().count(Adapter) != 0;
  }
  char Buf[256];
  int N;
  if (!Live) {
    // Pointer is NOT a currently-live adapter: stale/freed. Do NOT dereference.
    N = std::snprintf(Buf, sizeof(Buf),
                      "MANAGED_RELEASE adapter=%p resource=%p LIVE=0 "
                      "STALE_OR_FREED\n",
                      Adapter, Resource);
  } else {
    // Safe to inspect: report adapterReleased flag and first func-ptr word.
    const adapter_impl *A = static_cast<const adapter_impl *>(Adapter);
    const void *FirstFn = nullptr;
    static_assert(sizeof(void *) <= sizeof(A->UrFuncPtrs), "");
    std::memcpy(&FirstFn, &A->UrFuncPtrs, sizeof(FirstFn));
    N = std::snprintf(Buf, sizeof(Buf),
                      "MANAGED_RELEASE adapter=%p resource=%p LIVE=1 "
                      "adapterReleased=%d firstFuncPtr=%p\n",
                      Adapter, Resource, A->adapterReleased ? 1 : 0, FirstFn);
  }
  if (N > 0)
    diagAppend(Buf, static_cast<size_t>(N));
}

void adapter_impl::ur_failed_throw_exception(sycl::errc errc,
                                             ur_result_t ur_result) const {
  assert(ur_result != UR_RESULT_SUCCESS);
  std::string message =
      __SYCL_UR_ERROR_REPORT(MBackend) + codeToString(ur_result);

  if (ur_result == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
    assert(!adapterReleased);
    const char *last_error_message = nullptr;
    int32_t adapter_error = 0;
    ur_result = call_nocheck<UrApiKind::urAdapterGetLastError>(
        MAdapter, &last_error_message, &adapter_error);
    if (last_error_message)
      message += "\n" + std::string(last_error_message) + "(adapter error )" +
                 std::to_string(adapter_error) + "\n";
  }

  throw set_ur_error(sycl::exception(sycl::make_error_code(errc), message),
                     ur_result);
}

} // namespace detail
} // namespace _V1
} // namespace sycl

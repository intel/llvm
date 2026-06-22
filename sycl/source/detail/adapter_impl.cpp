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

#include <cstdlib>
#include <fstream>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
// DbgHelp gives us the loaded module base so raw return addresses can be turned
// into module-relative addresses (RVAs) for offline symbolization with the PDB.
#include <psapi.h>
#else
#include <execinfo.h>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

// DIAGNOSTIC (intel/llvm#22367). See declaration in adapter_impl.hpp.
void traceManagedAdapterExpired(void *Resource) {
  const char *Path = std::getenv("SYCL_TRACE_ADAPTER_UAF");
  if (!Path || !*Path)
    return;

  // Serialize writers; ~Managed can run on multiple threads during teardown.
  static std::mutex Mtx;
  std::lock_guard<std::mutex> Lock(Mtx);

  std::ofstream OS(Path, std::ios::app);
  if (!OS)
    return;

  OS << "MANAGED_ADAPTER_EXPIRED resource=" << Resource << "\n";

#ifdef _WIN32
  // Capture up to 62 frames (the documented max for a single call) and report
  // them as offsets from the module that contains each address, so they can be
  // symbolized later with: llvm-symbolizer --obj=<sycl dll> <rva>.
  void *Frames[62];
  USHORT N = RtlCaptureStackBackTrace(0, 62, Frames, nullptr);
  for (USHORT I = 0; I < N; ++I) {
    HMODULE Mod = nullptr;
    uintptr_t Rva = 0;
    // Full on-disk path of the module, so a symbolizer step on the runner can
    // resolve the RVA directly: llvm-symbolizer --obj=<ModPath> <rva>.
    char ModPath[MAX_PATH] = {0};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(Frames[I]), &Mod)) {
      Rva = reinterpret_cast<uintptr_t>(Frames[I]) -
            reinterpret_cast<uintptr_t>(Mod);
      GetModuleFileNameA(Mod, ModPath, sizeof(ModPath));
    }
    OS << "  frame " << I << " abs=" << Frames[I] << " rva=0x" << std::hex << Rva
       << std::dec << " module=" << ModPath << "\n";
  }
#else
  void *Frames[64];
  int N = backtrace(Frames, 64);
  char **Syms = backtrace_symbols(Frames, N);
  for (int I = 0; I < N; ++I)
    OS << "  frame " << I << " " << (Syms ? Syms[I] : "") << "\n";
  std::free(Syms);
#endif
  OS << "END\n";
  OS.flush();
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

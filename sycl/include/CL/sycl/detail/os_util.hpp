//===-- os_util.hpp - OS utilities -----------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Abstracts the operating system services.

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#define __SYCL_RT_OS_WINDOWS
// Windows platform
#ifdef _WIN64
// 64-bit Windows platform
#else
// 32-bit Windows platform
#endif // _WIN64
#elif __linux__
// Linux platform
#define __SYCL_RT_OS_LINUX
#define __SYCL_RT_OS_POSIX_SUPPORT
#elif defined(__APPLE__) && defined(__MACH__)
// Apple OSX
#define __SYCL_RT_OS_DARWIN
#define __SYCL_RT_OS_POSIX_SUPPORT
#else
#error "Unsupported compiler or OS"
#endif // _WIN32

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/// Uniquely identifies an operating system module (executable or a dynamic
/// library)
using OSModuleHandle = intptr_t;

/// Groups the OS-dependent services.
class __SYCL_EXPORT OSUtil {
public:
  /// Returns a module enclosing given address or nullptr.
  static OSModuleHandle getOSModuleHandle(const void *VirtAddr);

  /// Returns an absolute path to a directory where the object was found.
  static std::string getCurrentDSODir();

  /// Returns a directory component of a path.
  static std::string getDirName(const char *Path);

  /// Module handle for the executable module - it is assumed there is always
  /// single one at most.
  static constexpr OSModuleHandle ExeModuleHandle = -1;

  /// Dummy module handle to designate non-existing module for a device binary
  /// image loaded from file e.g. via SYCL_USE_KERNEL_SPV env var.
  static constexpr OSModuleHandle DummyModuleHandle = -2;

#ifdef __SYCL_RT_OS_WINDOWS
  static constexpr const char *DirSep = "\\";
#else
  static constexpr const char *DirSep = "/";
#endif

  /// Returns the amount of RAM available for the operating system.
  static size_t getOSMemSize();

  /// Allocates \p NumBytes bytes of uninitialized storage whose alignment
  /// is specified by \p Alignment.
  static void *alignedAlloc(size_t Alignment, size_t NumBytes);

  /// Deallocates the memory referenced by \p Ptr.
  static void alignedFree(void *Ptr);

  /// Make directory recursively and returns zero code on success
  static int makeDir(const char *Dir);

  /// Checks if specified path is present
  static inline bool isPathPresent(const std::string &Path) {
#ifdef __SYCL_RT_OS_WINDOWS
    struct _stat Stat;
    return !_stat(Path.c_str(), &Stat);
#else
    struct stat Stat;
    return !stat(Path.c_str(), &Stat);
#endif
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

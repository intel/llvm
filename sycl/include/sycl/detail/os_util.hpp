//===-- os_util.hpp - OS utilities -----------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Abstracts the operating system services.

#pragma once

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT

#include <cstdlib>    // for size_t
#include <string>     // for string
#include <sys/stat.h> // for stat

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

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Groups the OS-dependent services.
class __SYCL_EXPORT OSUtil {
public:
  /// Returns an absolute path to a directory where the object was found.
  static std::string getCurrentDSODir();

  /// Returns a directory component of a path.
  static std::string getDirName(const char *Path);

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

  /// Make all directories on the path, throws on error.
  static int makeDir(const char *Dir);

  /// Checks if specified path is present
  static bool isPathPresent(const std::string &Path) {
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
} // namespace _V1
} // namespace sycl

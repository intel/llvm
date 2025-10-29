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

#include <array>
#include <cstdlib> // for size_t
#include <functional>
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
#if !defined(__INTEL_PREVIEW_BREAKING_CHANGES)
#ifdef _WIN32
  // Access control is part of the mangling on Windows, have to preserve this
  // for backward ABI compatibility.
public:
#endif
  /// Returns a directory component of a path.
  static std::string getDirName(const char *Path);
#endif

public:
  /// Returns an absolute path to a directory where the object was found.
#if defined(__INTEL_PREVIEW_BREAKING_CHANGES)
  __SYCL_DLL_LOCAL
#endif
  static std::string getCurrentDSODir();

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

  /// Checks if specified path is present.
  static bool isPathPresent(const std::string &Path) {
#ifdef __SYCL_DEVICE_ONLY__
    (void)Path;
    return false;
#elif defined(__SYCL_RT_OS_WINDOWS)
    struct _stat Stat;
    return !_stat(Path.c_str(), &Stat);
#else
    struct stat Stat;
    return !stat(Path.c_str(), &Stat);
#endif
  }
};

// These functions are not a part of OSUtils class to prevent
// exporting them as ABI. They are only used in persistent cache
// implementation and should not be exposed to the end users.
// Get size of directory in bytes.
size_t getDirectorySize(const std::string &Path, bool ignoreErrors = false);

// Get size of file in bytes.
size_t getFileSize(const std::string &Path);

// Function to recursively iterate over the directory and execute
// 'Func' on each regular file.
void fileTreeWalk(const std::string Path,
                  std::function<void(const std::string)> Func,
                  bool ignoreErrors = false);

// Look up a function name that was dynamically linked
// This is used by the runtime where it needs to manipulate native handles
// (e.g. retaining OpenCL handles).
//
// The library must already have been loaded (perhaps by UR), otherwise this
// function throws a SYCL runtime exception.
void *dynLookup(const char *const *LibNames, size_t LibNameSizes,
                const char *FunName);

template <typename fn>
fn *dynLookupFunction(const char *const *LibNames, size_t LibNameSize,
                      const char *FunName) {
  return reinterpret_cast<fn *>(dynLookup(LibNames, LibNameSize, FunName));
}

// On Linux, first try to load from libur_adapter_opencl.so, then
// libur_adapter_opencl.so.0 if the first is not found. libur_adapter_opencl.so
// and libur_adapter_opencl.so.0 might be different libraries if they are not
// symlinked, which is the case with PyPi compiler distribution package.
// We can't load libur_adapter_opencl.so.0 always as the first choice because
// that would break SYCL unittests, which rely on mocking libur_adapter_opencl.
#ifdef __SYCL_RT_OS_WINDOWS
constexpr std::array<const char *, 1> OCLLibNames = {"OpenCL"};
#else
constexpr std::array<const char *, 2> OCLLibNames = {
    "libur_adapter_opencl.so", "libur_adapter_opencl.so.0"};
#endif

#define __SYCL_OCL_CALL(FN, ...)                                               \
  (sycl::_V1::detail::dynLookupFunction<decltype(FN)>(                         \
      sycl::detail::OCLLibNames.data(), sycl::detail::OCLLibNames.size(),      \
      #FN)(__VA_ARGS__))

} // namespace detail
} // namespace _V1
} // namespace sycl

//===-- os_util.cpp - OS utilities implementation---------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/os_util.hpp>

#include <cassert>
#include <limits>

#if __GNUC__ && __GNUC__ < 8
// Don't include <filesystem> for GCC versions less than 8
#else
#include <filesystem> // C++ 17 std::create_directories
#endif

#if defined(__SYCL_RT_OS_LINUX)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <libgen.h> // for dirname
#include <link.h>
#include <linux/limits.h> // for PATH_MAX
#include <sys/stat.h>
#include <sys/sysinfo.h>

#elif defined(__SYCL_RT_OS_WINDOWS)

#include <detail/windows_os_utils.hpp>

#include <Windows.h>
#include <malloc.h>
#include <shlwapi.h>

#elif defined(__SYCL_RT_OS_DARWIN)

#include <dlfcn.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#endif // __SYCL_RT_OS

namespace sycl {
inline namespace _V1 {
namespace detail {

#if defined(__SYCL_RT_OS_LINUX)
bool procMapsAddressInRange(std::istream &Stream, uintptr_t Addr) {
  uintptr_t Start = 0, End = 0;
  Stream >> Start;
  assert(!Stream.fail() && Stream.peek() == '-' &&
         "Couldn't read /proc/self/maps correctly");
  Stream.ignore(1);

  Stream >> End;
  assert(!Stream.fail() && Stream.peek() == ' ' &&
         "Couldn't read /proc/self/maps correctly");
  Stream.ignore(1);

  return Addr >= Start && Addr < End;
}

/// Returns an absolute path to a directory where the object was found.
std::string OSUtil::getCurrentDSODir() {
  // Examine /proc/self/maps and find where this function (getCurrendDSODir)
  // comes from - this is supposed to be an absolute path to libsycl.so.
  //
  // File structure is the following:
  //   address           perms offset  dev   inode       pathname
  //   00400000-00452000 r-xp 00000000 08:02 173521      /usr/bin/foo
  //   007c2000-007c8000 r--p 001c2000 fc:05 52567930    /usr/bin/bar
  //
  // We need to:
  //
  //  1) Iterate over lines and find the line which have an address of the
  //     current function in an `address' range.
  //
  //  2) Check that perms have read and executable flags (since we do execute
  //     this function).
  //
  //  3) Skip offset, dev, inode
  //
  //  4) Extract an absolute path to a filename and get a dirname from it.
  //
  uintptr_t CurrentFunc = (uintptr_t)&getCurrentDSODir;
  std::ifstream Stream("/proc/self/maps");
  Stream >> std::hex;
  while (!Stream.eof()) {
    if (!procMapsAddressInRange(Stream, CurrentFunc)) {
      // Skip the rest until an EOL and check the next line
      Stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    char Perm[4];
    Stream.readsome(Perm, sizeof(Perm));
    assert(Perm[0] == 'r' && Perm[2] == 'x' &&
           "Invalid flags in /proc/self/maps");
    assert(Stream.peek() == ' ');
    Stream.ignore(1);

    // Read and ignore the following:
    // offset
    Stream.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    Stream.ignore(1);
    // dev major
    Stream.ignore(std::numeric_limits<std::streamsize>::max(), ':');
    Stream.ignore(1);
    // dev minor
    Stream.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    Stream.ignore(1);
    // inode
    Stream.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    Stream.ignore(1);

    // Now read the path: it is padded with whitespaces, so we skip them
    // first.
    while (Stream.peek() == ' ') {
      Stream.ignore(1);
    }
    char Path[PATH_MAX];
    Stream.getline(Path, PATH_MAX - 1);
    Path[PATH_MAX - 1] = '\0';
    return OSUtil::getDirName(Path);
  }
  assert(false && "Unable to find the current function in /proc/self/maps");
  return "";
}

std::string OSUtil::getDirName(const char *Path) {
  std::string Tmp(Path);
  // dirname(3) needs a writable C string: a null-terminator is written where a
  // path should split.
  size_t TruncatedSize = strlen(dirname(const_cast<char *>(Tmp.c_str())));
  Tmp.resize(TruncatedSize);
  return Tmp;
}

#elif defined(__SYCL_RT_OS_WINDOWS)

/// Returns an absolute path where the object was found.
//  pi_win_proxy_loader.dll uses this same logic. If it is changed
//  significantly, it might be wise to change it there too.
std::string OSUtil::getCurrentDSODir() {
  char Path[MAX_PATH];
  Path[0] = '\0';
  Path[sizeof(Path) - 1] = '\0';
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODir));
  DWORD Ret = GetModuleFileNameA(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle),
      reinterpret_cast<LPSTR>(&Path), sizeof(Path));
  assert(Ret < sizeof(Path) && "Path is longer than PATH_MAX?");
  assert(Ret > 0 && "GetModuleFileNameA failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpecA(reinterpret_cast<LPSTR>(&Path));
  assert(RetCode && "PathRemoveFileSpecA failed");
  (void)RetCode;

  return Path;
}

std::string OSUtil::getDirName(const char *Path) {
  std::string Tmp(Path);
  // Remove trailing directory separators
  Tmp.erase(Tmp.find_last_not_of("/\\") + 1, std::string::npos);

  size_t pos = Tmp.find_last_of("/\\");
  if (pos != std::string::npos)
    return Tmp.substr(0, pos);

  // If no directory separator is present return initial path like dirname does
  return Tmp;
}

#elif defined(__SYCL_RT_OS_DARWIN)
std::string OSUtil::getCurrentDSODir() {
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

#endif // __SYCL_RT_OS

size_t OSUtil::getOSMemSize() {
#if defined(__SYCL_RT_OS_LINUX)
  struct sysinfo MemInfo;
  sysinfo(&MemInfo);
  return static_cast<size_t>(MemInfo.totalram * MemInfo.mem_unit);
#elif defined(__SYCL_RT_OS_WINDOWS)
  MEMORYSTATUSEX MemInfo;
  MemInfo.dwLength = sizeof(MemInfo);
  GlobalMemoryStatusEx(&MemInfo);
  return static_cast<size_t>(MemInfo.ullTotalPhys);
#elif defined(__SYCL_RT_OS_DARWIN)
  int64_t Size = 0;
  sysctlbyname("hw.memsize", &Size, nullptr, nullptr, 0);
  return static_cast<size_t>(Size);
#endif // __SYCL_RT_OS
}

void *OSUtil::alignedAlloc(size_t Alignment, size_t NumBytes) {
#if defined(__SYCL_RT_OS_LINUX) && (defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) ||    \
                                    defined(_LIBCPP_HAS_C11_FEATURES))
  return aligned_alloc(Alignment, NumBytes);
#elif defined(__SYCL_RT_OS_POSIX_SUPPORT)
  void *Addr = nullptr;
  int ReturnCode = posix_memalign(&Addr, Alignment, NumBytes);
  return (ReturnCode == 0) ? Addr : nullptr;
#elif defined(__SYCL_RT_OS_WINDOWS)
  return _aligned_malloc(NumBytes, Alignment);
#endif
}

void OSUtil::alignedFree(void *Ptr) {
#if defined(__SYCL_RT_OS_LINUX) || defined(__SYCL_RT_OS_POSIX_SUPPORT)
  free(Ptr);
#elif defined(__SYCL_RT_OS_WINDOWS)
  _aligned_free(Ptr);
#endif
}

// Make all directories on the path, throws on error.
int OSUtil::makeDir(const char *Dir) {
  assert((Dir != nullptr) && "Passed null-pointer as directory name.");
  if (isPathPresent(Dir))
    return 0;

// older GCC doesn't have full C++ 17 support.
#if __GNUC__ && __GNUC__ < 8
  std::string Path{Dir}, CurPath;
  size_t pos = 0;

  do {
    pos = Path.find_first_of("/\\", ++pos);
    CurPath = Path.substr(0, pos);
#if defined(__SYCL_RT_OS_POSIX_SUPPORT)
    auto Res = mkdir(CurPath.c_str(), 0777);
#else
    auto Res = _mkdir(CurPath.c_str());
#endif
    if (Res && errno != EEXIST)
      throw std::runtime_error("Failed to mkdir: " + CurPath + " (" +
                               std::strerror(errno) + ")");

  } while (pos != std::string::npos);
#else
  // using filesystem is simpler, more reliable, works better on Win
  std::filesystem::path path(Dir);
  std::filesystem::create_directories(path.make_preferred());
#endif
  return 0;
}

} // namespace detail
} // namespace _V1
} // namespace sycl

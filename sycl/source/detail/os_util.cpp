//===-- os_util.cpp - OS utilities implementation---------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/exception.hpp>

#include <cassert>

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

#include <Windows.h>
#include <direct.h>
#include <malloc.h>
#include <shlwapi.h>

#elif defined(__SYCL_RT_OS_DARWIN)

#include <dlfcn.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#endif // __SYCL_RT_OS

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#if defined(__SYCL_RT_OS_LINUX)

struct ModuleInfo {
  const void *VirtAddr; // in
  void *Handle;         // out
  const char *Name;     // out
};

constexpr OSModuleHandle OSUtil::ExeModuleHandle;
constexpr OSModuleHandle OSUtil::DummyModuleHandle;

static int callback(struct dl_phdr_info *Info, size_t, void *Data) {
  auto Base = reinterpret_cast<unsigned char *>(Info->dlpi_addr);
  auto MI = reinterpret_cast<ModuleInfo *>(Data);
  auto TestAddr = reinterpret_cast<const unsigned char *>(MI->VirtAddr);

  for (int i = 0; i < Info->dlpi_phnum; ++i) {
    unsigned char *SegStart = Base + Info->dlpi_phdr[i].p_vaddr;
    unsigned char *SegEnd = SegStart + Info->dlpi_phdr[i].p_memsz;

    // check if the tested address is within current segment
    if (TestAddr >= SegStart && TestAddr < SegEnd) {
      // ... it is - belongs to the module then
      // dlpi_addr is zero for the executable, replace it
      auto H = reinterpret_cast<void *>(Info->dlpi_addr);
      MI->Handle = H ? H : reinterpret_cast<void *>(OSUtil::ExeModuleHandle);
      MI->Name = Info->dlpi_name;
      return 1; // non-zero tells to finish iteration via modules
    }
  }
  return 0;
}

OSModuleHandle OSUtil::getOSModuleHandle(const void *VirtAddr) {
  ModuleInfo Res = {VirtAddr, nullptr, nullptr};
  dl_iterate_phdr(callback, &Res);

  return reinterpret_cast<OSModuleHandle>(Res.Handle);
}

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
  uintptr_t CurrentFunc = (uintptr_t) &getCurrentDSODir;
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

std::string OSUtil::getDirName(const char* Path) {
  std::string Tmp(Path);
  // dirname(3) needs a writable C string: a null-terminator is written where a
  // path should split.
  size_t TruncatedSize = strlen(dirname(const_cast<char *>(Tmp.c_str())));
  Tmp.resize(TruncatedSize);
  return Tmp;
}

#elif defined(__SYCL_RT_OS_WINDOWS)
OSModuleHandle OSUtil::getOSModuleHandle(const void *VirtAddr) {
  HMODULE PhModule;
  DWORD Flag = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  auto LpModuleAddr = reinterpret_cast<LPCSTR>(VirtAddr);
  if (!GetModuleHandleExA(Flag, LpModuleAddr, &PhModule)) {
    // Expect the caller to check for zero and take
    // necessary action
    return 0;
  }
  if (PhModule == GetModuleHandleA(nullptr))
    return OSUtil::ExeModuleHandle;
  return reinterpret_cast<OSModuleHandle>(PhModule);
}

/// Returns an absolute path where the object was found.
std::string OSUtil::getCurrentDSODir() {
  char Path[MAX_PATH];
  Path[0] = '\0';
  Path[sizeof(Path) - 1] = '\0';
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODir));
  DWORD Ret = GetModuleFileNameA(
      reinterpret_cast<HMODULE>(OSUtil::ExeModuleHandle == Handle ? 0 : Handle),
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
OSModuleHandle OSUtil::getOSModuleHandle(const void *VirtAddr) {
  Dl_info Res;
  dladdr(VirtAddr, &Res);
  return reinterpret_cast<OSModuleHandle>(Res.dli_fbase);
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

/* This is temporary solution until std::filesystem is available when SYCL RT
 * is moved to c++17 standard*/

/* Create directory recursively and return non zero code on success*/
int OSUtil::makeDir(const char *Dir) {
  assert((Dir != nullptr) && "Passed null-pointer as directory name.");
  if (isPathPresent(Dir))
    return 0;

  std::string Path{Dir}, CurPath;
  size_t pos = 0;

  do {
    pos = Path.find_first_of("/\\", ++pos);
    CurPath = Path.substr(0, pos);
#if defined(__SYCL_RT_OS_LINUX)
    auto Res = mkdir(CurPath.c_str(), 0777);
#else
    auto Res = _mkdir(CurPath.c_str());
#endif
    if (Res && errno != EEXIST)
      return Res;
  } while (pos != std::string::npos);
  return 0;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

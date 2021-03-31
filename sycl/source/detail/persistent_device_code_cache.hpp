//==---------- persistent_cache.hpp - On-disk cache for program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/device_binary_image.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/* This is temporary solution until std::filesystem is available when SYCL RT
 * is moved to c++17 standard*/
std::string getDirName(const char *Path);

#include <sys/stat.h>
/// Checks if specified path is present
inline bool isPathPresent(const std::string &Path) {
  struct stat Stat;
  return !stat(Path.c_str(), &Stat);
}

int makeDir(const char *Dir);

class LockCacheItem {
  const std::string FileName;

public:
  LockCacheItem(const std::string &DirName) : FileName(DirName + "/.lock") {
    int fd;
    while ((fd = open(FileName.c_str(), O_CREAT | O_EXCL, S_IWRITE)) == -1) {
      std::this_thread::yield();
    }
    close(fd);
  }
  static bool isLocked(const std::string &DirName) {
    return isPathPresent(DirName + "/.lock");
  }
  ~LockCacheItem() { std::remove(FileName.c_str()); }
};

class PersistentDeviceCodeCache {
private:
  /* Write built binary to persistent cache
   * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
   */
  static void writeBinaryDataToFile(const std::string &FileName,
                                    const std::vector<std::vector<char>> &Data);

  /* Read built binary to persistent cache
   * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
   */
  static std::vector<std::vector<char>>
  readBinaryDataFromFile(const std::string &FileName);

  /* Writing cache item key sources to be used for reliable identification
   * Format: Four pairs of [size, value] for device, build options,
   * specialization constant values, device code SPIR-V image.
   */
  static void writeSourceItem(const std::string &FileName, const device &Device,
                              const RTDeviceBinaryImage &Img,
                              const SerializedObj &SpecConsts,
                              const std::string &BuildOptionsString);

  /* Check that cache item key sources are equal to the current program
   */
  static bool isCacheItemSrcEqual(const std::string &FileName,
                                  const device &Device,
                                  const RTDeviceBinaryImage &Img,
                                  const SerializedObj &SpecConsts,
                                  const std::string &BuildOptionsString);

  /* Check if on-disk cache enabled.
   */
  static bool isEnabled();

  /* Returns the path to directory storing persistent device code cache.*/
  static std::string getRootDir();

  /* Form string representing device version */
  static std::string getDeviceIDString(const device &Device);

public:
  /* Get directory name for storing current cache item
   */
  static std::string getCacheItemPath(const device &Device,
                                      const RTDeviceBinaryImage &Img,
                                      const SerializedObj &SpecConsts,
                                      const std::string &BuildOptionsString);

  /* Program binaries built for one or more devices are read from persistent
   * cache and returned in form of vector of programs. Each binary program is
   * stored in vector of chars.
   */
  static std::vector<std::vector<char>>
  getItemFromDisc(const device &Device, const RTDeviceBinaryImage &Img,
                  const SerializedObj &SpecConsts,
                  const std::string &BuildOptionsString,
                  RT::PiProgram &NativePrg);

  /* Stores build program in persisten cache
   */
  static void putItemToDisc(const device &Device,
                            const RTDeviceBinaryImage &Img,
                            const SerializedObj &SpecConsts,
                            const std::string &BuildOptionsString,
                            const RT::PiProgram &NativePrg);
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

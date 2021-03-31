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
/*class LockCacheItem {
  const std::string FileName;

public:
  LockCacheItem(const std::string &DirName) : FileName(DirName + ".lock") {
    int fd;
    while ((fd = open(FileName.c_str(), O_CREAT | O_EXCL, S_IWRITE)) == -1) {
      std::this_thread::yield();
    }
    close(fd);
  }
  ~LockCacheItem() { std::remove(FileName.c_str()); }
};
bool lockCacheDir(const std::string &DirName) {
  int fd = open((DirName+".lock").c_str(), O_CREAT | O_EXCL, S_IWRITE);
  if(fd!=-1){
    close(fd);
    return true;
  }
  return false;
}
void unlockCacheDir(const std::string &DirName) {
  std::remove(DirName+".lock");
}*/

class PersistentCache {

  /* Form string representing device version */
  static std::string getDeviceString(const device &Device);
  /* Write built binary to persistent cache
   * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
   */
  static void writeCacheItemBin(const std::string &FileName,
                                const std::vector<std::vector<char>> &Data);
  /* Read built binary to persistent cache
   * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
   */
  static std::vector<std::vector<char>>
  readCacheItem(const std::string &FileName);
  /* Writing cache item key sources to be used for reliable identification
   * Format: Four pairs of [size, value] for device, build options,
   * specialization constant values, device code SPIR-V image.
   */
  static void writeCacheItemSrc(const std::string &FileName,
                                const device &Device,
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
  /* Get directory name for storing current cache item
   */
  static std::string getCacheItemDirName(const device &Device,
                                         const RTDeviceBinaryImage &Img,
                                         const SerializedObj &SpecConsts,
                                         const std::string &BuildOptionsString);
  /* Check if on-disk cache enabled.
   */
  static bool isPersistentCacheEnabled();

  /* Returns the path to directory storing persistent device code cache.*/
  static std::string getDeviceCodeCacheRoot();

public:
  /* Program binaries built for one or more devices are read from persistent
   * cache and returned in form of vector of programs. Each binary program is
   * stored in vector of chars.
   */
  static std::vector<std::vector<char>>
  getPIProgramFromDisc(const device &Device, const RTDeviceBinaryImage &Img,
                       const SerializedObj &SpecConsts,
                       const std::string &BuildOptionsString,
                       RT::PiProgram &NativePrg);
  /* Stores build program in persisten cache
   */
  static void putPIProgramToDisc(const detail::plugin &Plugin,
                                 const device &Device,
                                 const RTDeviceBinaryImage &Img,
                                 const SerializedObj &SpecConsts,
                                 const std::string &BuildOptionsString,
                                 const RT::PiProgram &Program);
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

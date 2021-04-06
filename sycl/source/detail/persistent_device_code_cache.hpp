//==---------- persistent_device_code_cache.hpp -----------------*- C++-*---==//
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
#include <chrono>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/* This is temporary solution until std::filesystem is available when SYCL RT
 * is moved to c++17 standard*/
std::string getDirName(const char *Path);

/// Checks if specified path is present
inline bool isPathPresent(const std::string &Path) {
  struct stat Stat;
  return !stat(Path.c_str(), &Stat);
}

/// Checks if file age exceeds defined threshold
bool exceedLifeTime(const std::string &Path, time_t sec);

/// Make directory recursibely
int makeDir(const char *Dir);

class LockCacheItem {
private:
  const std::string FileName;
  bool Owned = false;
  static const char LockSuffix[];

public:
  LockCacheItem(const std::string &Path);

  bool isOwned() { return Owned; }
  static bool isLocked(const std::string &Path) {
    return isPathPresent(Path + LockSuffix);
  }
  ~LockCacheItem();
};
/* End of temporary solution*/

class PersistentDeviceCodeCache {
  /* The device code images are stored on file system using structure below:
   * <cache_root>/
   *     <device_hash>/
   *         <device_image_hash>/
   *             <spec_constants_values_hash>/
   *                 <build_options_hash>/
   *                     <n>.src
   *                     <n>.bin
   *                     .lock
   *   <cache_root>                 - root directory storing cache files;
   *   <device_hash>                - hash out of device information used to
   *                                  identify target device;
   *   <device_image_hash>          - hash made out of device image used as
   * input for the JIT compilation; <spec_constants_values_hash> - hash for
   * specialization constants values; <build_options_hash>         - hash for
   * all build options; <n>                          - sequential number of hash
   * collisions. When hashes matches for the specific build but full values
   * don't, new cache item is added with incremented value (enumeration started
   *                                  from 0).
   * Two files per cache item are stored on disk:
   *   <n>.src - contains full values for build parameters (device information,
   *             specialization constant values, build options, device image)
   *             which is used to resolve hash collisions and analysis of cached
   *             items.
   *   <n>.bin - contains built device code.
   * Also directory lock file is created when cache item is written. Lock item
   *   .lock   - directory lock file. It is created when data is save to
   *             filesystem. On read operation the absence of file is checked
   *             but not created to avoid lock.
   * All filesystem operations are not treated as SYCL errors and ignored. If
   * such errors happen warning messages are written to std::err:
   *  - on cache write operation cache item is not created;
   *  - on cache read operation it is treated as cache miss.
   *
   * To avoid concurent write operations to the same cache item causing data
   * corruption cache item dir is locked using .lock file. It is created on
   * write operation and checked on read.
   *  - Lock is done per cache item.
   *  - Lock is not blocking. If lock fails cache item read/write is skipped
   *  and SYCL application flow resumes. There is time threshold for locking
   *  a cache item: 10 microseconds.
   *  - If lock file exists for 1 hour it is cleared on next access to resume
   *  caching for the particular cache item.
   */
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

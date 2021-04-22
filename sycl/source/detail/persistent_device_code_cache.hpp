//==---------- persistent_device_code_cache.hpp -----------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/device_binary_image.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <detail/config.hpp>
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

/* The class manages inter-process synchronization:
 *  - Path passed to the constructor is appended with .lock and used as lock
 *    file.
 *  - All operations are not blocking and failure ignoring (diagnostic may be
 *    sent to std::cerr when SYCL_CACHE_TRACE environment variable is set).
 *  - There are two modes of accessing shared resource:
 *    - write access assumes that lock is acquired (object is created and
 *      isOwned() method confirms that current executor owns the lock);
 *    - read access checks that the lock is not acquired for write by others
 *      with the help of isLocked() method.
 */
class LockCacheItem {
private:
  const std::string FileName;
  bool Owned = false;
  static const char LockSuffix[];

public:
  LockCacheItem(const std::string &Path);

  bool isOwned() { return Owned; }
  static bool isLocked(const std::string &Path) {
    return OSUtil::isPathPresent(Path + LockSuffix);
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
   *                                  input for the JIT compilation;
   *   <spec_constants_values_hash> - hash for specialization constants values;
   *   <build_options_hash>         - hash for all build options;
   *   <n>                          - sequential number of hash collisions.
   *                                  When hashes match for the specific build
   *                                  but full values don't, new cache item is
   *                                  added with incremented value(enumeration
   *                                  started from 0).
   * Two files per cache item are stored on disk:
   *   <n>.src  - contains full values for build parameters (device information,
   *              specialization constant values, build options, device image)
   *              which is used to resolve hash collisions and analysis of
   *              cached items.
   *   <n>.bin  - contains built device code.
   *   <n>.lock - cache item lock file. It is created when data is saved to
   *              filesystem. On read operation the absence of file is checked
   *              but it is not created to avoid lock.
   * All filesystem operation failures are not treated as SYCL errors and
   * ignored. If such errors happen warning messages are written to std::cerr
   * and:
   *  - on cache write operation cache item is not created;
   *  - on cache read operation it is treated as cache miss.
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

  /* Returns true if specified image should be cached on disk. It checks if
   * cache is enabled, image has SPIRV type and matches thresholds. */
  static bool isImageCached(const RTDeviceBinaryImage &Img);

  /* Returns value of specified parameter. Default value is used if failure
   * happens during obtaining value. */
  template <ConfigID Config>
  static unsigned long getNumParam(unsigned long Default) {
    auto Value = SYCLConfig<Config>::get();
    try {
      if (Value)
        return std::stol(Value);
    } catch (std::exception const &) {
      PersistentDeviceCodeCache::trace("Invalid value provided, use default " +
                                       std::to_string(Default));
    }
    return Default;
  }

  /* Default value for minimum device code size to be cached on disk in bytes */
  static constexpr unsigned long DEFAULT_MIN_DEVICE_IMAGE_SIZE = 0;

  /* Default value for maximum device code size to be cached on disk in bytes */
  static constexpr unsigned long DEFAULT_MAX_DEVICE_IMAGE_SIZE =
      1024 * 1024 * 1024;

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
                  const std::string &BuildOptionsString);

  /* Stores build program in persisten cache
   */
  static void putItemToDisc(const device &Device,
                            const RTDeviceBinaryImage &Img,
                            const SerializedObj &SpecConsts,
                            const std::string &BuildOptionsString,
                            const RT::PiProgram &NativePrg);

  /* Sends message to std:cerr stream when SYCL_CACHE_TRACE environemnt is set*/
  static void trace(const std::string &msg) {
    static const char *TraceEnabled = SYCLConfig<SYCL_CACHE_TRACE>::get();
    if (TraceEnabled)
      std::cerr << "*** Code caching: " << msg << std::endl;
  }
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

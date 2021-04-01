//==---------- persistent_cache.cpp - On-disk cache for program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/plugin.hpp>
#if defined(__SYCL_RT_OS_LINUX)
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/* This is temporary solution until std::filesystem is available when SYCL RT
 * is moved to c++17 standard*/
std::string getDirName(const char *Path) {
  std::string Tmp(Path);
  // Remove trailing directory separators
  Tmp.erase(Tmp.find_last_not_of("/\\") + 1, std::string::npos);

  auto pos = Tmp.find_last_of("/\\");
  if (pos != std::string::npos)
    return Tmp.substr(0, pos);

  // If no directory separator is present return initial path like dirname does
  return Tmp;
}

int makeDir(const char *Dir) {
  assert((Dir != nullptr) && "Passed null-pointer as directory name.");
  // Directory is present - do nothing
  if (isPathPresent(Dir))
    return 0;

  char *CurDir = strdup(Dir);
  makeDir(getDirName(CurDir).c_str());

  free(CurDir);

#if defined(__SYCL_RT_OS_LINUX)
  return mkdir(Dir, 0777);
#else
  return _mkdir(Dir);
#endif
}

LockCacheItem::LockCacheItem(const std::string &DirName)
    : FileName(DirName + "/.lock") {
  int fd;
  while ((fd = open(FileName.c_str(), O_CREAT | O_EXCL, S_IWRITE)) == -1) {
    std::this_thread::yield();
  }
  close(fd);
}

/* Stores build program in persisten cache
 */
void PersistentDeviceCodeCache::putItemToDisc(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString,
    const RT::PiProgram &NativePrg) {

  if (!isEnabled())
    return;

  // Only SPIRV images are cached
  if (Img.getFormat() != PI_DEVICE_BINARY_TYPE_SPIRV)
    return;

  auto Plugin = detail::getSyclObjImpl(Device)->getPlugin();
  std::string DirName =
      getCacheItemPath(Device, Img, SpecConsts, BuildOptionsString);

  size_t i = 0;
  std::string FileName;
  do {
    FileName = DirName + "/" + std::to_string(i++);
  } while (isPathPresent(FileName + ".bin"));

  unsigned int DeviceNum = 0;

  Plugin.call<PiApiKind::piProgramGetInfo>(
      NativePrg, PI_PROGRAM_INFO_NUM_DEVICES, sizeof(DeviceNum), &DeviceNum,
      nullptr);

  std::vector<size_t> BinarySizes(DeviceNum);
  Plugin.call<PiApiKind::piProgramGetInfo>(
      NativePrg, PI_PROGRAM_INFO_BINARY_SIZES,
      sizeof(size_t) * BinarySizes.size(), BinarySizes.data(), nullptr);

  std::vector<std::vector<char>> Result;
  std::vector<char *> Pointers;
  for (size_t I = 0; I < BinarySizes.size(); ++I) {
    Result.emplace_back(BinarySizes[I]);
    Pointers.push_back(Result[I].data());
  }

  Plugin.call<PiApiKind::piProgramGetInfo>(NativePrg, PI_PROGRAM_INFO_BINARIES,
                                           sizeof(char *) * Pointers.size(),
                                           Pointers.data(), nullptr);

  try {
    makeDir(DirName.c_str());
    LockCacheItem Lock{DirName};
    writeBinaryDataToFile(FileName + ".bin", Result);
    writeSourceItem(FileName + ".src", Device, Img, SpecConsts,
                    BuildOptionsString);
  } catch (...) {
    // If a problem happens on storing cache item, do nothing
  }
}

/* Program binaries built for one or more devices are read from persistent
 * cache and returned in form of vector of programs. Each binary program is
 * stored in vector of chars.
 */
std::vector<std::vector<char>> PersistentDeviceCodeCache::getItemFromDisc(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString,
    RT::PiProgram &NativePrg) {

  if (!isEnabled())
    return {};

  // Only SPIRV images are cached
  if (Img.getFormat() != PI_DEVICE_BINARY_TYPE_SPIRV)
    return {};

  std::string Path =
      getCacheItemPath(Device, Img, SpecConsts, BuildOptionsString);

  if (!isPathPresent(Path))
    return {};

  int i = 0;

  // If cache directory is locked ignore cache
  if (LockCacheItem::isLocked(Path))
    return {};

  std::string FileName{Path + "/" + std::to_string(i)};
  while (isPathPresent(FileName + ".bin") || isPathPresent(FileName + ".src")) {
    if (isCacheItemSrcEqual(FileName + ".src", Device, Img, SpecConsts,
                            BuildOptionsString)) {
      try {
        return readBinaryDataFromFile(FileName + ".bin");
      } catch (...) {
        // If read was unsuccessfull try the next item
      }
    }
    FileName = Path + "/" + std::to_string(++i);
  }
  return {};
}

/* Returns string value which can be used to identify different device
 */
std::string PersistentDeviceCodeCache::getDeviceIDString(const device &Device) {
  return Device.get_platform().get_info<sycl::info::platform::name>() + "/" +
         Device.get_info<sycl::info::device::name>() + "/" +
         Device.get_info<sycl::info::device::version>() + "/" +
         Device.get_info<sycl::info::device::driver_version>();
}

/* Write built binary to persistent cache
 * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
 * Return on first unsuccessfull file operation
 */
void PersistentDeviceCodeCache::writeBinaryDataToFile(
    const std::string &FileName, const std::vector<std::vector<char>> &Data) {
  std::ofstream FileStream{FileName, std::ios::binary};
  if (FileStream.fail())
    return;

  size_t Size = Data.size();
  FileStream.write((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return;

  for (size_t i = 0; i < Data.size(); ++i) {
    Size = Data[i].size();
    FileStream.write((char *)&Size, sizeof(Size));
    if (FileStream.fail())
      return;
    FileStream.write(Data[i].data(), Size);
    if (FileStream.fail())
      return;
  }
  FileStream.close();
}

/* Read built binary to persistent cache
 * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
 */
std::vector<std::vector<char>>
PersistentDeviceCodeCache::readBinaryDataFromFile(const std::string &FileName) {
  std::ifstream FileStream{FileName, std::ios::binary};
  if (FileStream.fail())
    return {};
  size_t ImgNum, ImgSize;
  FileStream.read((char *)&ImgNum, sizeof(ImgNum));
  if (FileStream.fail())
    return {};

  std::vector<std::vector<char>> Res(ImgNum);
  for (size_t i = 0; i < ImgNum; ++i) {
    FileStream.read((char *)&ImgSize, sizeof(ImgSize));
    if (FileStream.fail())
      return {};

    std::vector<char> ImgData(ImgSize);
    FileStream.read(ImgData.data(), ImgSize);
    if (FileStream.fail())
      return {};

    Res[i] = std::move(ImgData);
  }

  return Res;
}

/* Writing cache item key sources to be used for reliable identification
 * Format: Four pairs of [size, value] for device, build options, specialization
 * constant values, device code SPIR-V image.
 */
void PersistentDeviceCodeCache::writeSourceItem(
    const std::string &FileName, const device &Device,
    const RTDeviceBinaryImage &Img, const SerializedObj &SpecConsts,
    const std::string &BuildOptionsString) {
  std::ofstream FileStream{FileName, std::ios::binary};
  if (FileStream.fail())
    return;

  std::string DeviceString{getDeviceIDString(Device)};

  size_t Size = DeviceString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return;
  FileStream.write(DeviceString.data(), Size);
  if (FileStream.fail())
    return;
  Size = BuildOptionsString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return;
  FileStream.write(BuildOptionsString.data(), Size);
  if (FileStream.fail())
    return;
  Size = SpecConsts.size();
  FileStream.write((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return;
  FileStream.write((const char *)SpecConsts.data(), Size);
  if (FileStream.fail())
    return;
  Size = Img.getSize();
  FileStream.write((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return;
  FileStream.write((const char *)Img.getRawData().BinaryStart, Size);
  if (FileStream.fail())
    return;
  FileStream.close();
}

/* Check that cache item key sources are equal to the current program.
 * If file read operations fail cache item is treated as not equal.
 */
bool PersistentDeviceCodeCache::isCacheItemSrcEqual(
    const std::string &FileName, const device &Device,
    const RTDeviceBinaryImage &Img, const SerializedObj &SpecConsts,
    const std::string &BuildOptionsString) {
  std::ifstream FileStream{FileName, std::ios::binary};
  if (FileStream.fail())
    return false;

  std::string ImgString{(const char *)Img.getRawData().BinaryStart,
                        Img.getSize()};
  std::string DeviceString{getDeviceIDString(Device)};
  std::string SpecConstsString{(const char *)SpecConsts.data(),
                               SpecConsts.size()};

  size_t Size;
  FileStream.read((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return false;

  std::string res(Size, '\0');
  FileStream.read(&res[0], Size);
  if (FileStream.fail() || DeviceString.compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return false;

  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (FileStream.fail() || BuildOptionsString.compare(0, Size, res.data()))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return false;

  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (FileStream.fail() || SpecConstsString.compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  if (FileStream.fail())
    return false;

  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (FileStream.fail() || ImgString.compare(res))
    return false;

  FileStream.close();
  return true;
}

/* Returns directory name to store specific kernel image for specified
 * device, build options and specialization constants values.
 */
std::string PersistentDeviceCodeCache::getCacheItemPath(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  static std::string cache_root{getRootDir()};

  std::string ImgString{(const char *)Img.getRawData().BinaryStart,
                        Img.getSize()};
  std::string DeviceString{getDeviceIDString(Device)};
  std::string SpecConstsString{(const char *)SpecConsts.data(),
                               SpecConsts.size()};
  std::hash<std::string> StringHasher{};

  return cache_root + "/" + std::to_string(StringHasher(DeviceString)) + "/" +
         std::to_string(StringHasher(ImgString)) + "/" +
         std::to_string(StringHasher(SpecConstsString)) + "/" +
         std::to_string(StringHasher(BuildOptionsString));
}

/* Returns true if persistent cache enabled. The cache can be disabled by
 * setting SYCL_CACHE_EVICTION_DISABLE environmnet variable.
 */
bool PersistentDeviceCodeCache::isEnabled() {
  static const char *PersistenCacheDisabled =
      SYCLConfig<SYCL_CACHE_DISABLE_PERSISTENT>::get();
  return !PersistenCacheDisabled;
}

/* Returns path for device code cache root directory
 */
std::string PersistentDeviceCodeCache::getRootDir() {
  static const char *RootDir = SYCLConfig<SYCL_CACHE_DIR>::get();
  if (RootDir)
    return RootDir;

  constexpr char DeviceCodeCacheDir[] = "/libsycl_cache";

  // Use static to calculate directory only once per program run
#if defined(__SYCL_RT_OS_LINUX)
  static const char *CacheDir = std::getenv("XDG_CACHE_HOME");
  static const char *HomeDir = std::getenv("HOME");
  static std::string Res{
      std::string(CacheDir
                      ? CacheDir
                      : (HomeDir ? std::string(HomeDir) + "/.cache" : ".")) +
      DeviceCodeCacheDir};
#else
  static const char *AppDataDir = std::getenv("AppData");
  static std::string Res{std::string(AppDataDir ? AppDataDir : ".") +
                         DeviceCodeCacheDir};
#endif
  return Res;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

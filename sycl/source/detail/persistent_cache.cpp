//==---------- persistent_cache.cpp - On-disk cache for program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/persistent_cache.hpp>
#include <detail/plugin.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// These are temporary implementation of file operations until moving to C++17
// and use of std::filesystem instead 

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

#include <sys/stat.h>
/// Checks if specified path is present
static inline bool isPathPresent(const std::string &Path) {
   struct stat Stat;
   return !stat(Path.c_str(), &Stat);
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


/* Stores build program in persisten cache
 */
void PersistentCache::putPIProgramToDisc(const detail::plugin &Plugin,
                                         const device &Device,
                                         const RTDeviceBinaryImage &Img,
                                         const SerializedObj &SpecConsts,
                                         const std::string &BuildOptionsString,
                                         const RT::PiProgram &Program) {

  if (!isPersistentCacheEnabled())
    return;

  // Only SPIRV images are cached
  if (Img.getFormat() != PI_DEVICE_BINARY_TYPE_SPIRV &&
      (Img.getFormat() == PI_DEVICE_BINARY_TYPE_NONE &&
       pi::getBinaryImageFormat(Img.getRawData().BinaryStart, Img.getSize()) !=
           PI_DEVICE_BINARY_TYPE_SPIRV))
    return;

  std::string DirName =
      getCacheItemDirName(Device, Img, SpecConsts, BuildOptionsString);

  size_t i = 0;
  std::string FileName;
  do {
    FileName = DirName + "/" + std::to_string(i++);
  } while (isPathPresent(FileName + ".bin"));

  unsigned int DeviceNum = 0;

  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_NUM_DEVICES,
                                           sizeof(DeviceNum), &DeviceNum,
                                           nullptr);

  std::vector<size_t> BinarySizes(DeviceNum);
  Plugin.call<PiApiKind::piProgramGetInfo>(
      Program, PI_PROGRAM_INFO_BINARY_SIZES,
      sizeof(size_t) * BinarySizes.size(), BinarySizes.data(), nullptr);

  std::vector<std::vector<char>> Result;
  std::vector<char *> Pointers;
  for (size_t I = 0; I < BinarySizes.size(); ++I) {
    Result.emplace_back(BinarySizes[I]);
    Pointers.push_back(Result[I].data());
  }

  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_BINARIES,
                                           sizeof(char *) * Pointers.size(),
                                           Pointers.data(), nullptr);

  makeDir(DirName.c_str());
  writeCacheItemBin(FileName + ".bin", Result);
  writeCacheItemSrc(FileName + ".src", Device, Img, SpecConsts,
                    BuildOptionsString);
}

/* Program binaries built for one or more devices are read from persistent
 * cache and returned in form of vector of programs. Each binary program is
 * stored in vector of chars.
 */
std::vector<std::vector<char>> PersistentCache::getPIProgramFromDisc(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString,
    RT::PiProgram &NativePrg) {

  if (!isPersistentCacheEnabled())
    return {};

  // Only SPIRV images are cached
  if (Img.getFormat() != PI_DEVICE_BINARY_TYPE_SPIRV &&
      (Img.getFormat() == PI_DEVICE_BINARY_TYPE_NONE &&
       pi::getBinaryImageFormat(Img.getRawData().BinaryStart, Img.getSize()) !=
           PI_DEVICE_BINARY_TYPE_SPIRV))
    return {};

  std::string Path =
      getCacheItemDirName(Device, Img, SpecConsts, BuildOptionsString);

  if (!isPathPresent(Path))
    return {};

  int i = 0;
  std::string FileName{Path + "/" + std::to_string(i)};
  while (isPathPresent(FileName + ".bin") ||
         isPathPresent(FileName + ".src")) {
    if (isCacheItemSrcEqual(FileName + ".src", Device, Img, SpecConsts,
                            BuildOptionsString)) {
      return readCacheItem(FileName + ".bin");
    }
    FileName = Path + "/" + std::to_string(++i);
  }
  return {};
}

/* Returns string value which can be used to identify different device
 */
std::string PersistentCache::getDeviceString(const device &Device) {
  return Device.get_platform().get_info<sycl::info::platform::name>() + "/" +
         Device.get_info<sycl::info::device::name>() + "/" +
         Device.get_info<sycl::info::device::version>() + "/" +
         Device.get_info<sycl::info::device::driver_version>();
}

/* Write built binary to persistent cache
 * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
 */
void PersistentCache::writeCacheItemBin(
    const std::string &FileName, const std::vector<std::vector<char>> &Data) {
  std::ofstream FileStream{FileName, std::ios::binary};

  size_t Size = Data.size();
  FileStream.write((char *)&Size, sizeof(Size));
  for (size_t i = 0; i < Data.size(); ++i) {
    Size = Data[i].size();
    FileStream.write((char *)&Size, sizeof(Size));
    FileStream.write(Data[i].data(), Size);
  }
  FileStream.close();
}

/* Read built binary to persistent cache
 * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
 */
std::vector<std::vector<char>>
PersistentCache::readCacheItem(const std::string &FileName) {
  std::ifstream FileStream{FileName, std::ios::binary};
  size_t ImgNum, ImgSize;
  FileStream.read((char *)&ImgNum, sizeof(ImgNum));
  std::vector<std::vector<char>> Res(ImgNum);
  for (size_t i = 0; i < ImgNum; ++i) {
    FileStream.read((char *)&ImgSize, sizeof(ImgSize));
    std::vector<char> ImgData(ImgSize);
    FileStream.read(ImgData.data(), ImgSize);
    Res[i] = std::move(ImgData);
  }

  return Res;
}

/* Writing cache item key sources to be used for reliable identification
 * Format: Four pairs of [size, value] for device, build options, specialization
 * constant values, device code SPIR-V image.
 */
void PersistentCache::writeCacheItemSrc(const std::string &FileName,
                                        const device &Device,
                                        const RTDeviceBinaryImage &Img,
                                        const SerializedObj &SpecConsts,
                                        const std::string &BuildOptionsString) {
  std::ofstream FileStream{FileName, std::ios::binary};
  std::string DeviceString{getDeviceString(Device)};

  size_t Size = DeviceString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(DeviceString.data(), Size);
  Size = BuildOptionsString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(BuildOptionsString.data(), Size);
  Size = SpecConsts.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write((const char *)SpecConsts.data(), Size);
  Size = Img.getSize();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write((const char *)Img.getRawData().BinaryStart, Size);
  FileStream.close();
}

/* Check that cache item key sources are equal to the current program
 */
bool PersistentCache::isCacheItemSrcEqual(
    const std::string &FileName, const device &Device,
    const RTDeviceBinaryImage &Img, const SerializedObj &SpecConsts,
    const std::string &BuildOptionsString) {
  std::ifstream FileStream{FileName, std::ios::binary};
  std::string ImgString{(const char *)Img.getRawData().BinaryStart,
                        Img.getSize()};
  std::string DeviceString{getDeviceString(Device)};
  std::string SpecConstsString{(const char *)SpecConsts.data(),
                               SpecConsts.size()};

  size_t Size;
  std::string res;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (DeviceString.compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (BuildOptionsString.compare(0, Size, res.data()))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (SpecConstsString.compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (ImgString.compare(res))
    return false;

  FileStream.close();
  return true;
}

/* Returns directory name to store specific kernel image for specified
 * device, build options and specialization constants values.
 */
std::string PersistentCache::getCacheItemDirName(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  static std::string cache_root{getDeviceCodeCacheRoot()};

  std::string ImgString{(const char *)Img.getRawData().BinaryStart,
                        Img.getSize()};
  std::string DeviceString{getDeviceString(Device)};
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
bool PersistentCache::isPersistentCacheEnabled() {
  static const char *PersistenCacheDisabled =
      SYCLConfig<SYCL_CACHE_DISABLE_PERSISTENT>::get();
  return !PersistenCacheDisabled;
}

/* Returns path for device code cache root directory
 */
std::string PersistentCache::getDeviceCodeCacheRoot() {
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

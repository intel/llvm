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
void PersistentCache::putPIProgramToDisc(const detail::plugin &Plugin,
                                         const device &Device,
                                         const RTDeviceBinaryImage &Img,
                                         const SerializedObj &SpecConsts,
                                         const std::string &BuildOptionsString,
                                         const RT::PiProgram &Program) {
  // Only SPIRV images are cached
  if (Img.getFormat() != PI_DEVICE_BINARY_TYPE_SPIRV &&
      (Img.getFormat() == PI_DEVICE_BINARY_TYPE_NONE &&
       pi::getBinaryImageFormat(Img.getRawData().BinaryStart, Img.getSize()) !=
           PI_DEVICE_BINARY_TYPE_SPIRV))
    return;

  if (!isPersistentCacheEnabled()) {
    return;
  }

  std::string DirName =
      getCacheItemDirName(Device, Img, SpecConsts, BuildOptionsString);

  size_t i = 0;
  std::string FileName;
  do {
    FileName = DirName + "/" + std::to_string(i++);
  } while (OSUtil::isPathPresent(FileName + ".bin"));

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

  OSUtil::makeDir(DirName.c_str());
  writeCacheItemBin(FileName + ".bin", Result);
  writeCacheItemSrc(FileName + ".src", Device, Img, SpecConsts,
                    BuildOptionsString);
}

std::vector<std::vector<char>> PersistentCache::getPIProgramFromDisc(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString,
    RT::PiProgram &NativePrg) {

  // Only SPIRV images are cached
  if (Img.getFormat() != PI_DEVICE_BINARY_TYPE_SPIRV &&
      (Img.getFormat() == PI_DEVICE_BINARY_TYPE_NONE &&
       pi::getBinaryImageFormat(Img.getRawData().BinaryStart, Img.getSize()) !=
           PI_DEVICE_BINARY_TYPE_SPIRV))
    return {};

  if (!isPersistentCacheEnabled())
    return {};

  std::string Path{
      getCacheItemDirName(Device, Img, SpecConsts, BuildOptionsString)};

  if (!OSUtil::isPathPresent(Path))
    return {};

  int i = 0;
  std::string FileName{Path + "/" + std::to_string(i)};
  while (OSUtil::isPathPresent(FileName + ".bin") &&
         OSUtil::isPathPresent(FileName + ".src")) {
    if (isCacheItemSrcEqual(FileName + ".src", Device, Img, SpecConsts,
                            BuildOptionsString)) {
      return readCacheItem(FileName + ".bin");
    }
    FileName = Path + "/" + std::to_string(++i);
  }

  return {};
}

std::string PersistentCache::getDeviceString(const device &Device) {
  return {Device.get_platform().get_info<sycl::info::platform::name>() + "/" +
          Device.get_info<sycl::info::device::name>() + "/" +
          Device.get_info<sycl::info::device::version>() + "/" +
          Device.get_info<sycl::info::device::driver_version>()};
}

std::string PersistentCache::dumpBinData(const unsigned char *Data,
                                         size_t Size) {
  if (!Size)
    return "NONE";
  std::stringstream ss;
  for (size_t i = 0; i < Size; i++) {
    ss << std::hex << (int)Data[i];
  }
  return ss.str();
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
  std::vector<std::vector<char>> Res;
  std::ifstream FileStream{FileName, std::ios::binary};
  size_t ImgNum, ImgSize;
  FileStream.read((char *)&ImgNum, sizeof(ImgNum));
  Res.resize(ImgNum);
  for (size_t i = 0; i < ImgNum; ++i) {
    FileStream.read((char *)&ImgSize, sizeof(ImgSize));
    Res[i].resize(ImgSize);
    FileStream.read(Res[i].data(), ImgSize);
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
  std::string ImgString{
      dumpBinData(Img.getRawData().BinaryStart, Img.getSize())};
  std::string DeviceString{getDeviceString(Device)};
  std::string SpecConstsString{
      dumpBinData(SpecConsts.data(), SpecConsts.size())};

  size_t Size = DeviceString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(DeviceString.data(), Size);
  Size = BuildOptionsString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(BuildOptionsString.data(), Size);
  Size = SpecConstsString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(SpecConstsString.data(), Size);
  Size = ImgString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(ImgString.data(), Size);
  FileStream.close();
}

/* Check that cache item key sources are equal to the current program
 */
bool PersistentCache::isCacheItemSrcEqual(
    const std::string &FileName, const device &Device,
    const RTDeviceBinaryImage &Img, const SerializedObj &SpecConsts,
    const std::string &BuildOptionsString) {
  std::ifstream FileStream{FileName, std::ios::binary};
  std::string ImgString{
      dumpBinData(Img.getRawData().BinaryStart, Img.getSize())};
  std::string DeviceString{getDeviceString(Device)};
  std::string SpecConstsString{
      dumpBinData(SpecConsts.data(), SpecConsts.size())};

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
  if (SpecConstsString.compare(0, Size, res.data()))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (ImgString.compare(0, Size, res.data()))
    return false;

  FileStream.close();
  return true;
}

std::string PersistentCache::getCacheItemDirName(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  static std::string cache_root{detail::OSUtil::getCacheRoot()};

  std::string ImgString{
      dumpBinData(Img.getRawData().BinaryStart, Img.getSize())};
  std::string DeviceString{getDeviceString(Device)};
  std::string SpecConstsString{
      dumpBinData(SpecConsts.data(), SpecConsts.size())};
  std::hash<std::string> StringHasher{};

  return {cache_root + "/" + std::to_string(StringHasher(DeviceString)) + "/" +
          std::to_string(StringHasher(ImgString)) + "/" +
          std::to_string(StringHasher(SpecConstsString)) + "/" +
          std::to_string(StringHasher(BuildOptionsString))};
}

bool PersistentCache::isPersistentCacheEnabled() {
  static const char *PersistenCacheDisabled =
      SYCLConfig<SYCL_CACHE_DISABLE_PERSISTENT>::get();
  return !PersistenCacheDisabled;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

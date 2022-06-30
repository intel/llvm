//==---------- persistent_device_code_cache.cpp -----------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <cstdio>
#include <optional>

#if defined(__SYCL_RT_OS_LINUX)
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/* Lock file suffix */
const char LockCacheItem::LockSuffix[] = ".lock";

LockCacheItem::LockCacheItem(const std::string &Path)
    : FileName(Path + LockSuffix) {
  int fd;

  /* If the lock fail is not created */
  if ((fd = open(FileName.c_str(), O_CREAT | O_EXCL, S_IWRITE)) != -1) {
    close(fd);
    Owned = true;
  } else {
    PersistentDeviceCodeCache::trace("Failed to aquire lock file: " + FileName);
  }
}

LockCacheItem::~LockCacheItem() {
  if (Owned && std::remove(FileName.c_str()))
    PersistentDeviceCodeCache::trace("Failed to release lock file: " +
                                     FileName);
}

// Returns true if the specified format is either SPIRV or a native binary.
static bool IsSupportedImageFormat(RT::PiDeviceBinaryType Format) {
  return Format == PI_DEVICE_BINARY_TYPE_SPIRV ||
         Format == PI_DEVICE_BINARY_TYPE_NATIVE;
}

/* Returns true if specified image should be cached on disk. It checks if
 * cache is enabled, image has supported format and matches thresholds. */
bool PersistentDeviceCodeCache::isImageCached(const RTDeviceBinaryImage &Img) {
  // Cache should be enabled and image type is one of the supported formats.
  if (!isEnabled() || !IsSupportedImageFormat(Img.getFormat()))
    return false;

  // Disable cache for ITT-profiled images.
  if (SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::get()) {
    return false;
  }

  // TODO: Move parsing logic and caching to specializations of SYCLConfig.
  static auto MaxImgSize = getNumParam<SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE>(
      DEFAULT_MAX_DEVICE_IMAGE_SIZE);
  static auto MinImgSize = getNumParam<SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE>(
      DEFAULT_MIN_DEVICE_IMAGE_SIZE);

  // Make sure that image size is between caching thresholds if they are set.
  // Zero values for threshold is treated as disabled threshold.
  if ((MaxImgSize && (Img.getSize() > MaxImgSize)) ||
      (MinImgSize && (Img.getSize() < MinImgSize)))
    return false;

  return true;
}

/* Stores built program in persisten cache
 */
void PersistentDeviceCodeCache::putItemToDisc(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString,
    const RT::PiProgram &NativePrg) {

  if (!isImageCached(Img))
    return;

  std::string DirName =
      getCacheItemPath(Device, Img, SpecConsts, BuildOptionsString);

  if (DirName.empty())
    return;

  auto Plugin = detail::getSyclObjImpl(Device)->getPlugin();

  size_t i = 0;
  std::string FileName;
  do {
    FileName = DirName + "/" + std::to_string(i++);
  } while (OSUtil::isPathPresent(FileName + ".bin"));

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
    OSUtil::makeDir(DirName.c_str());
    LockCacheItem Lock{FileName};
    if (Lock.isOwned()) {
      std::string FullFileName = FileName + ".bin";
      writeBinaryDataToFile(FullFileName, Result);
      trace("device binary has been cached: " + FullFileName);
      writeSourceItem(FileName + ".src", Device, Img, SpecConsts,
                      BuildOptionsString);
      FILE* dummy= fopen(FullFileName.c_str(),"rb");
      fprintf(stderr,"Dummy write open:%p\n",dummy);
      fclose(dummy);
    }
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
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {

  if (!isImageCached(Img))
    return {};

  std::string Path =
      getCacheItemPath(Device, Img, SpecConsts, BuildOptionsString);

  if (Path.empty() || !OSUtil::isPathPresent(Path))
    return {};

  int i = 0;

  std::string FileName{Path + "/" + std::to_string(i)};
  while (OSUtil::isPathPresent(FileName + ".bin") &&
         OSUtil::isPathPresent(FileName + ".src")) {

    if (!LockCacheItem::isLocked(FileName) &&
        isCacheItemSrcEqual(FileName + ".src", Device, Img, SpecConsts,
                            BuildOptionsString)) {
      try {
      // FILE* dummy= fopen((FileName +".bin").c_str(),"rb");
      // fprintf(stderr,"Dummy read open:%p\n",dummy);
      // fclose(dummy);
        std::string FullFileName = FileName + ".bin";
        std::vector<std::vector<char>> res =
            readBinaryDataFromFile(FullFileName);
        trace("using cached device binary: " + FullFileName);
        return res; // subject for NRVO
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
  // FILE* file=fopen(FileName.c_str(),"wb");
  // fprintf(stderr,"writeBinaryDataToFile filepath:%s file:%p\n ",FileName.c_str(),file);
  size_t Size = Data.size();
  fwrite(&Size,sizeof(Size),1,file);
  // fprintf(stderr,"writeBinaryDataToFile size:%ld \n",Size);

  for (size_t i = 0; i < Data.size(); ++i) {
    Size = Data[i].size();
    fwrite(&Size,sizeof(Size),1,file);
    fwrite(Data[i].data(),sizeof(char),Size,file);
    // fprintf(stderr,"writeBinaryDataToFile size loop:%ld %p \n",Size,Data[i].data());
  }
  fclose(file);

  if (ferror(file))
    trace("Failed to write binary file " + FileName);
}

/* Read built binary to persistent cache
 * Format: numImages, 1stImageSize, Image[, NthImageSize, NthImage...]
 */
std::vector<std::vector<char>>
PersistentDeviceCodeCache::readBinaryDataFromFile(const std::string &FileName) {
  FILE* file= fopen(FileName.c_str(),"rb");
  // if(file == nullptr){
  //   perror("Error reading file from readBinaryData:");
  // }
  size_t ImgNum = 0, ImgSize = 0;
  // fprintf(stderr,"readBinaryDataFromFile filepath:%s file:%p\n ",FileName.c_str(),file);
  fread(&ImgNum,sizeof(ImgNum),1,file);
  // fprintf(stderr,"readBinaryDataFromFile size:%ld \n",ImgNum);

  std::vector<std::vector<char>> Res(ImgNum);
  for (size_t i = 0; i < ImgNum; ++i) {
    fread(&ImgSize,sizeof(ImgSize),1,file);

    std::vector<char> ImgData(ImgSize);
    fread(ImgData.data(),sizeof(char),ImgSize,file);
    // fprintf(stderr,"readBinaryDataFromFile size loop:%ld %p \n",ImgSize,ImgData.data());
    Res[i] = std::move(ImgData);
  }
  fclose(file);

  if (ferror(file) && !feof(file)) {
    trace("Failed to read binary file from " + FileName);
    return {};
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

  FILE* file= fopen(FileName.c_str(),"wb");

  std::string DeviceString{getDeviceIDString(Device)};

  size_t Size = DeviceString.size();
  fwrite(&Size,sizeof(Size),1,file);
  fwrite(DeviceString.data(),sizeof(char),Size,file);

  Size = BuildOptionsString.size();
  fwrite(&Size,sizeof(Size),1,file);
  fwrite(BuildOptionsString.data(),sizeof(char),Size,file);

  Size = SpecConsts.size();
  fwrite(&Size,sizeof(Size),1,file);
  fwrite(SpecConsts.data(),sizeof(SpecConsts.data()[0]),Size,file);

  Size = Img.getSize();
  fwrite(&Size,sizeof(Size),1,file);
  fwrite(Img.getRawData().BinaryStart,sizeof(Img.getRawData().BinaryStart[0]),Size,file);
  fclose(file);

  if (ferror(file)) {
    trace("Failed to write source file to " + FileName);
  }
}

/* Check that cache item key sources are equal to the current program.
 * If file read operations fail cache item is treated as not equal.
 */
bool PersistentDeviceCodeCache::isCacheItemSrcEqual(
    const std::string &FileName, const device &Device,
    const RTDeviceBinaryImage &Img, const SerializedObj &SpecConsts,
    const std::string &BuildOptionsString) {
  FILE* file= fopen(FileName.c_str(),"rb");
  // if(file == nullptr){
  //   perror("Error reading file from is CacheItem:");
  // }
  std::string ImgString{(const char *)Img.getRawData().BinaryStart,
                        Img.getSize()};
  std::string SpecConstsString{(const char *)SpecConsts.data(),
                               SpecConsts.size()};

  size_t Size = 0;
  fread(&Size,sizeof(Size),1,file);
  std::string res(Size, '\0');
  fread(&res[0],sizeof(char),Size,file);
  if (getDeviceIDString(Device).compare(res))
    return false;

  fread(&Size, sizeof(Size),1,file);
  res.resize(Size);
  fread(&res[0],sizeof(char),Size,file);
  if (BuildOptionsString.compare(res))
    return false;

  fread(&Size, sizeof(Size),1,file);
  res.resize(Size);
  fread(&res[0],sizeof(char),Size,file);
  if (SpecConstsString.compare(res))
    return false;

  fread(&Size, sizeof(Size),1,file);
  res.resize(Size);
  fread(&res[0],sizeof(char),Size,file);
  if (ImgString.compare(res))
    return false;

  fclose(file);

  if (ferror(file) && !feof(file)) {
    trace("Failed to read source file from " + FileName);
  }

  return true;
}

/* Returns directory name to store specific kernel image for specified
 * device, build options and specialization constants values.
 */
std::string PersistentDeviceCodeCache::getCacheItemPath(
    const device &Device, const RTDeviceBinaryImage &Img,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  std::string cache_root{getRootDir()};
  if (cache_root.empty()) {
    trace("Disable persistent cache due to unconfigured cache root.");
    return {};
  }

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

/* Returns true if persistent cache is enabled.
 */
bool PersistentDeviceCodeCache::isEnabled() {
  bool CacheIsEnabled = SYCLConfig<SYCL_CACHE_PERSISTENT>::get();
  static bool FirstCheck = true;
  if (FirstCheck) {
    PersistentDeviceCodeCache::trace(CacheIsEnabled ? "enabled" : "disabled");
    FirstCheck = false;
  }
  return CacheIsEnabled;
}

/* Returns path for device code cache root directory
 */
std::string PersistentDeviceCodeCache::getRootDir() {
  return SYCLConfig<SYCL_CACHE_DIR>::get();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

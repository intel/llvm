//==---------- persistent_device_code_cache.cpp -----------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter.hpp>
#include <detail/device_impl.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <optional>

#if defined(__SYCL_RT_OS_POSIX_SUPPORT)
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>
#endif

namespace sycl {
inline namespace _V1 {
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
    PersistentDeviceCodeCache::trace("Failed to acquire lock file: " +
                                     FileName + " " + std::strerror(errno));
    PersistentDeviceCodeCache::trace("Failed to acquire lock file: " +
                                     FileName + " " + std::strerror(errno));
  }
}

LockCacheItem::~LockCacheItem() {
  if (Owned && std::remove(FileName.c_str()))
    PersistentDeviceCodeCache::trace("Failed to release lock file: ", FileName);
}

// Returns true if the specified format is either SPIRV or a native binary.
static bool IsSupportedImageFormat(ur::DeviceBinaryType Format) {
  return Format == SYCL_DEVICE_BINARY_TYPE_SPIRV ||
         Format == SYCL_DEVICE_BINARY_TYPE_NATIVE;
}

/* Returns true if specified images should be cached on disk. It checks if
 * cache is enabled, images have supported format and match thresholds. */
bool PersistentDeviceCodeCache::areImagesCacheable(
    const std::vector<const RTDeviceBinaryImage *> &Imgs) {
  assert(!Imgs.empty());
  auto Format = Imgs[0]->getFormat();
  assert(std::all_of(Imgs.begin(), Imgs.end(),
                     [&Format](const RTDeviceBinaryImage *Img) {
                       return Img->getFormat() == Format;
                     }) &&
         "All images are expected to have the same format");
  // Cache should be enabled and image type is one of the supported formats.
  if (!isEnabled() || !IsSupportedImageFormat(Format))
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
  size_t TotalSize = 0;
  for (const RTDeviceBinaryImage *Img : Imgs)
    TotalSize += Img->getSize();
  if ((MaxImgSize && (TotalSize > MaxImgSize)) ||
      (MinImgSize && (TotalSize < MinImgSize)))
    return false;

  return true;
}

static std::vector<const RTDeviceBinaryImage *>
getSortedImages(const std::vector<const RTDeviceBinaryImage *> &Imgs) {
  std::vector<const RTDeviceBinaryImage *> SortedImgs = Imgs;
  std::sort(SortedImgs.begin(), SortedImgs.end(),
            [](const RTDeviceBinaryImage *A, const RTDeviceBinaryImage *B) {
              // All entry names are unique among these images, so comparing the
              // first ones is enough.
              return std::strcmp(A->getRawData().EntriesBegin->name,
                                 B->getRawData().EntriesBegin->name) < 0;
            });
  return SortedImgs;
}

// Utility function to get a non-yet-existing unique filename.
std::string getUniqueFilename(const std::string &base_name) {
  size_t i = 0;
  std::string filename = base_name + "/" + std::to_string(i++);
  while (OSUtil::isPathPresent(filename + ".bin") ||
         OSUtil::isPathPresent(filename + ".lock")) {
    filename = base_name + "/" + std::to_string(i++);
  }
  return filename;
}

/* Returns binary data for the UR program. There is a one-to-one
 * correspondence between the vector of programs returned from the function and
 * the input vector of devices.
 */
std::vector<std::vector<char>>
getProgramBinaryData(const ur_program_handle_t &NativePrg,
                     const std::vector<device> &Devices) {
  assert(!Devices.empty() && "At least one device is expected");
  // We expect all devices to be from the same platform/adpater.
  auto Adapter = detail::getSyclObjImpl(Devices[0])->getAdapter();
  unsigned int DeviceNum = 0;
  Adapter->call<UrApiKind::urProgramGetInfo>(
      NativePrg, UR_PROGRAM_INFO_NUM_DEVICES, sizeof(DeviceNum), &DeviceNum,
      nullptr);

  std::vector<ur_device_handle_t> URDevices(DeviceNum);
  Adapter->call<UrApiKind::urProgramGetInfo>(
      NativePrg, UR_PROGRAM_INFO_DEVICES,
      sizeof(ur_device_handle_t) * URDevices.size(), URDevices.data(), nullptr);

  std::vector<size_t> BinarySizes(DeviceNum);
  Adapter->call<UrApiKind::urProgramGetInfo>(
      NativePrg, UR_PROGRAM_INFO_BINARY_SIZES,
      sizeof(size_t) * BinarySizes.size(), BinarySizes.data(), nullptr);

  std::vector<std::vector<char>> Binaries;
  std::vector<char *> Pointers;
  for (size_t I = 0; I < BinarySizes.size(); ++I) {
    Binaries.emplace_back(BinarySizes[I]);
    Pointers.push_back(Binaries[I].data());
  }

  Adapter->call<UrApiKind::urProgramGetInfo>(
      NativePrg, UR_PROGRAM_INFO_BINARIES, sizeof(char *) * Pointers.size(),
      Pointers.data(), nullptr);

  // Select only binaries for the input devices preserving one to one
  // correpsondence.
  std::vector<std::vector<char>> Result(Devices.size());
  for (size_t DeviceIndex = 0; DeviceIndex < Devices.size(); DeviceIndex++) {
    auto DeviceIt = std::find_if(
        URDevices.begin(), URDevices.end(),
        [&Devices, &DeviceIndex](const ur_device_handle_t &URDevice) {
          return URDevice ==
                 detail::getSyclObjImpl(Devices[DeviceIndex])->getHandleRef();
        });
    assert(DeviceIt != URDevices.end() &&
           "Device is not associated with the program");
    auto URDeviceIndex = std::distance(URDevices.begin(), DeviceIt);
    Result[DeviceIndex] = std::move(Binaries[URDeviceIndex]);
  }

  // Return binaries correpsonding to the input devices.

  return Result;
}

// Save the current time in a file.
void PersistentDeviceCodeCache::saveCurrentTimeInAFile(std::string FileName) {
  // Lock the file to prevent concurrent writes.
  LockCacheItem Lock{FileName};
  if (Lock.isOwned()) {
    try {
      std::ofstream FileStream{FileName, std::ios::trunc};
      FileStream << std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count();
      FileStream.close();
    } catch (std::exception &e) {
      throw sycl::exception(make_error_code(errc::runtime),
                            "Failed to save current time in a file: " +
                                FileName + "\n" + std::string(e.what()));
    }
  }
}

// Check if cache_size.txt file is present in the cache root directory.
// If not, create it and populate it with the size of the cache directory.
void PersistentDeviceCodeCache::repopulateCacheSizeFile(
    const std::string &CacheRoot) {

  // No need to store cache size if eviction is disabled.
  if (!isEvictionEnabled())
    return;

  const std::string CacheSizeFileName = "cache_size.txt";
  const std::string CacheSizeFile = CacheRoot + "/" + CacheSizeFileName;

  // Create cache root, if it does not exist.
  try {
    if (!OSUtil::isPathPresent(CacheRoot))
      OSUtil::makeDir(CacheRoot.c_str());
  } catch (...) {
    throw sycl::exception(make_error_code(errc::runtime),
                          "Failed to create cache root directory: " +
                              CacheRoot);
  }

  // If the cache size file is not present, calculate the size of the cache size
  // directory and write it to the file.
  if (!OSUtil::isPathPresent(CacheSizeFile)) {
    PersistentDeviceCodeCache::trace(
        "Cache size file not present. Creating one.");

    // Take the lock to write the cache size to the file.
    {
      LockCacheItem Lock{CacheSizeFile};
      if (!Lock.isOwned()) {
        // If some other process is writing the cache size, do not write it.
        PersistentDeviceCodeCache::trace("Didnot create the cache size file. "
                                         "Some other process is creating one.");

        // Stall until the other process creates the file. Stalling is important
        // to prevent race between one process that's calculating the directory
        // size and another process that's trying to create a new cache entry.
        while (!OSUtil::isPathPresent(CacheSizeFile))
          continue;
      } else {
        // Calculate the size of the cache directory.
        // During directory size calculation, do not add anything
        // in the cache. Otherwise, we'll get a std::fs_error.
        size_t CacheSize = getDirectorySize(CacheRoot, /*Ignore Error*/ true);

        std::ofstream FileStream{CacheSizeFile};
        FileStream << CacheSize;
        FileStream.close();
        PersistentDeviceCodeCache::trace("Cache size file created.");
      }
    }
  }
}

void PersistentDeviceCodeCache::evictItemsFromCache(
    const std::string &CacheRoot, size_t CacheSize, size_t MaxCacheSize) {
  PersistentDeviceCodeCache::trace("Cache eviction triggered.");

  // EVict half of the cache.
  constexpr float HowMuchCacheToEvict = 0.5;

  // Create a file eviction_in_progress.lock to indicate that eviction is in
  // progress. This file is used to prevent two processes from evicting the
  // cache at the same time.
  LockCacheItem Lock{CacheRoot + EvictionInProgressFileSuffix};
  if (!Lock.isOwned()) {
    // If some other process is evicting the cache, return.
    PersistentDeviceCodeCache::trace(
        "Another process is evicting the cache. Returning.");
    return;
  }

  // Get the list of all files in the cache directory along with their last
  // modification time.
  std::vector<std::pair<uint64_t, std::string>> FilesWithAccessTime;

  auto CollectFileAccessTime = [&FilesWithAccessTime](const std::string File) {
    if (File.find(CacheEntryAccessTimeSuffix) != std::string::npos) {
      std::ifstream FileStream{File};
      uint64_t AccessTime;
      FileStream >> AccessTime;
      FilesWithAccessTime.push_back({AccessTime, File});
    }
  };

  // fileTreeWalk can throw if any new file is created or removed during the
  // iteration. Retry in that case. When eviction is in progress, we don't
  // insert any new item but processes can still read the cache. Reading from
  // cache can create/remove .lock file which can cause the exception.
  while (true) {
    try {
      fileTreeWalk(CacheRoot, CollectFileAccessTime);
      break;
    } catch (...) {
      FilesWithAccessTime.clear();
      // If the cache directory is removed during the iteration, retry.
      continue;
    }
  }

  // Sort the files in the cache directory based on their last access time.
  std::sort(FilesWithAccessTime.begin(), FilesWithAccessTime.end(),
            [](const std::pair<uint64_t, std::string> &A,
               const std::pair<uint64_t, std::string> &B) {
              return A.first < B.first;
            });

  // Evict files from the cache directory until the cache size is less than the
  // threshold.
  size_t CurrCacheSize = CacheSize;
  for (const auto &File : FilesWithAccessTime) {

    int pos = File.second.find(CacheEntryAccessTimeSuffix);
    const std::string FileNameWOExt = File.second.substr(0, pos);
    const std::string BinFile = FileNameWOExt + ".bin";
    const std::string SrcFile = FileNameWOExt + ".src";

    while (OSUtil::isPathPresent(BinFile) || OSUtil::isPathPresent(SrcFile)) {

      // Lock to prevent race between writer and eviction thread.
      LockCacheItem Lock{FileNameWOExt};
      if (Lock.isOwned()) {
        // Remove the file and subtract its size from the cache size.
        auto RemoveFileAndSubtractSize = [&CurrCacheSize](
                                             const std::string &FileName) {
          // If the file is not present, return.
          // Src file is not present inj kernel_compiler cache, we will
          // skip removing it.
          if (!OSUtil::isPathPresent(FileName))
            return;

          auto FileSize = getFileSize(FileName);
          if (std::remove(FileName.c_str())) {
            throw sycl::exception(make_error_code(errc::runtime),
                                  "Failed to evict cache entry: " + FileName);
          } else {
            PersistentDeviceCodeCache::trace("File removed: ", FileName);
            CurrCacheSize -= FileSize;
          }
        };

        // If removal fails due to a race, retry.
        // Races are rare, but can happen if another process is reading the
        // file. Locking down the entire cache and blocking all readers would be
        // inefficient.
        try {
          RemoveFileAndSubtractSize(SrcFile);
          RemoveFileAndSubtractSize(BinFile);
        } catch (...) {
          continue;
        }
      }
    }

    // If the cache size is less than the threshold, break.
    if (CurrCacheSize <= (size_t)(HowMuchCacheToEvict * MaxCacheSize))
      break;
  }

  // Update the cache size file with the new cache size.
  {
    const std::string CacheSizeFileName = "cache_size.txt";
    const std::string CacheSizeFile = CacheRoot + "/" + CacheSizeFileName;
    while (true) {
      LockCacheItem Lock{CacheSizeFile};
      if (!Lock.isOwned()) {
        // If some other process is writing the cache size, spin lock.
        continue;
      } else {
        std::fstream FileStream;
        FileStream.open(CacheSizeFile, std::ios::out | std::ios::trunc);
        FileStream << CurrCacheSize;
        FileStream.close();

        PersistentDeviceCodeCache::trace(
            "Updating the cache size file after eviction. New size: " +
            std::to_string(CurrCacheSize));
        break;
      }
    }
  }
}

// Update the cache size file and trigger cache eviction if needed.
void PersistentDeviceCodeCache::updateCacheFileSizeAndTriggerEviction(
    const std::string &CacheRoot, size_t ItemSize) {

  // No need to store cache size if eviction is disabled.
  if (!isEvictionEnabled())
    return;

  const std::string CacheSizeFileName = "cache_size.txt";
  const std::string CacheSizeFile = CacheRoot + "/" + CacheSizeFileName;
  size_t CurrentCacheSize = 0;
  // Read the cache size from the file.
  while (true) {
    LockCacheItem Lock{CacheSizeFile};
    if (!Lock.isOwned()) {
      // If some other process is writing the cache size, spin lock.
      continue;
    } else {
      PersistentDeviceCodeCache::trace("Updating the cache size file.");
      std::fstream FileStream;
      FileStream.open(CacheSizeFile, std::ios::in);

      // Read the cache size from the file;
      std::string line;
      if (std::getline(FileStream, line)) {
        CurrentCacheSize = std::stoull(line);
      }
      FileStream.close();

      CurrentCacheSize += ItemSize;

      // Write the updated cache size to the file.
      FileStream.open(CacheSizeFile, std::ios::out | std::ios::trunc);
      FileStream << CurrentCacheSize;
      FileStream.close();
      break;
    }
  }

  // Check if the cache size exceeds the threshold and trigger cache eviction if
  // needed.
  size_t MaxCacheSize = SYCLConfig<SYCL_CACHE_MAX_SIZE>::getProgramCacheSize();
  if (CurrentCacheSize > MaxCacheSize) {
    // Trigger cache eviction.
    evictItemsFromCache(CacheRoot, CurrentCacheSize, MaxCacheSize);
  }
}

/* Stores built program in persistent cache. We will put the binary for each
 * device in the list to a separate file.
 */
void PersistentDeviceCodeCache::putItemToDisc(
    const std::vector<device> &Devices,
    const std::vector<const RTDeviceBinaryImage *> &Imgs,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString,
    const ur_program_handle_t &NativePrg) {

  if (!areImagesCacheable(Imgs))
    return;

  repopulateCacheSizeFile(getRootDir());

  // Do not insert any new item if eviction is in progress.
  // Since evictions are rare, we can afford to spin lock here.
  const std::string EvictionInProgressFile =
      getRootDir() + EvictionInProgressFileSuffix;
  // Stall until the other process finishes eviction.
  while (OSUtil::isPathPresent(EvictionInProgressFile))
    continue;

  std::vector<const RTDeviceBinaryImage *> SortedImgs = getSortedImages(Imgs);
  auto BinaryData = getProgramBinaryData(NativePrg, Devices);

  // Total size of the item that we just wrote to the cache.
  size_t TotalSize = 0;
  for (size_t DeviceIndex = 0; DeviceIndex < Devices.size(); DeviceIndex++) {
    // If we don't have binary for the device, skip it.
    if (BinaryData[DeviceIndex].empty())
      continue;
    std::string DirName = getCacheItemPath(Devices[DeviceIndex], SortedImgs,
                                           SpecConsts, BuildOptionsString);

    if (DirName.empty())
      return;

    std::string FileName;
    try {
      OSUtil::makeDir(DirName.c_str());
      FileName = getUniqueFilename(DirName);
      LockCacheItem Lock{FileName};
      if (Lock.isOwned()) {
        std::string FullFileName = FileName + ".bin";
        writeBinaryDataToFile(FullFileName, BinaryData[DeviceIndex]);
        trace("device binary has been cached: ", FullFileName);
        writeSourceItem(FileName + ".src", Devices[DeviceIndex], SortedImgs,
                        SpecConsts, BuildOptionsString);

        // Update Total cache size after adding the new items.
        TotalSize += getFileSize(FileName + ".src");
        TotalSize += getFileSize(FileName + ".bin");

        saveCurrentTimeInAFile(FileName + CacheEntryAccessTimeSuffix);
      } else {
        PersistentDeviceCodeCache::trace("cache lock not owned ", FileName);
      }
    } catch (std::exception &e) {
      PersistentDeviceCodeCache::trace(
          std::string("exception encountered making persistent cache: ") +
          e.what());
    } catch (...) {
      PersistentDeviceCodeCache::trace(
          std::string("error outputting persistent cache: ") +
          std::strerror(errno));
    }
  }

  // Update the cache size file and trigger cache eviction if needed.
  if (TotalSize)
    updateCacheFileSizeAndTriggerEviction(getRootDir(), TotalSize);
}

void PersistentDeviceCodeCache::putCompiledKernelToDisc(
    const std::vector<device> &Devices, const std::string &BuildOptionsString,
    const std::string &SourceStr, const ur_program_handle_t &NativePrg) {

  repopulateCacheSizeFile(getRootDir());

  // Do not insert any new item if eviction is in progress.
  // Since evictions are rare, we can afford to spin lock here.
  const std::string EvictionInProgressFile =
      getRootDir() + EvictionInProgressFileSuffix;
  // Stall until the other process finishes eviction.
  while (OSUtil::isPathPresent(EvictionInProgressFile))
    continue;

  auto BinaryData = getProgramBinaryData(NativePrg, Devices);
  // Total size of the item that we are writing to the cache.
  size_t TotalSize = 0;

  for (size_t DeviceIndex = 0; DeviceIndex < Devices.size(); DeviceIndex++) {
    // If we don't have binary for the device, skip it.
    if (BinaryData[DeviceIndex].empty())
      continue;
    std::string DirName = getCompiledKernelItemPath(
        Devices[DeviceIndex], BuildOptionsString, SourceStr);

    try {
      OSUtil::makeDir(DirName.c_str());
      std::string FileName = getUniqueFilename(DirName);
      LockCacheItem Lock{FileName};
      if (Lock.isOwned()) {
        std::string FullFileName = FileName + ".bin";
        writeBinaryDataToFile(FullFileName, BinaryData[DeviceIndex]);
        PersistentDeviceCodeCache::trace_KernelCompiler(
            "binary has been cached: ", FullFileName);

        TotalSize += getFileSize(FullFileName);
        saveCurrentTimeInAFile(FileName + CacheEntryAccessTimeSuffix);
      } else {
        PersistentDeviceCodeCache::trace_KernelCompiler("cache lock not owned ",
                                                        FileName);
      }
    } catch (std::exception &e) {
      PersistentDeviceCodeCache::trace_KernelCompiler(
          std::string("exception encountered making cache: ") + e.what());
    } catch (...) {
      PersistentDeviceCodeCache::trace_KernelCompiler(
          std::string("error outputting cache: ") + std::strerror(errno));
    }
  }

  // Update the cache size file and trigger cache eviction if needed.
  if (TotalSize)
    updateCacheFileSizeAndTriggerEviction(getRootDir(), TotalSize);
}

/* Program binaries built for one or more devices are read from persistent
 * cache and returned in form of vector of programs. Each binary program is
 * stored in vector of chars. There is a one-to-one correspondence between
 * the vector of programs returned from the function and the input vector of
 * devices.
 */
std::vector<std::vector<char>> PersistentDeviceCodeCache::getItemFromDisc(
    const std::vector<device> &Devices,
    const std::vector<const RTDeviceBinaryImage *> &Imgs,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  assert(!Devices.empty());
  if (!areImagesCacheable(Imgs))
    return {};

  std::vector<const RTDeviceBinaryImage *> SortedImgs = getSortedImages(Imgs);
  std::vector<std::vector<char>> Binaries(Devices.size());
  std::string FileNames;
  for (size_t DeviceIndex = 0; DeviceIndex < Devices.size(); DeviceIndex++) {
    std::string Path = getCacheItemPath(Devices[DeviceIndex], SortedImgs,
                                        SpecConsts, BuildOptionsString);

    if (Path.empty() || !OSUtil::isPathPresent(Path))
      return {};

    int i = 0;

    std::string FileName{Path + "/" + std::to_string(i)};
    while (OSUtil::isPathPresent(FileName + ".bin") ||
           OSUtil::isPathPresent(FileName + ".src")) {

      if (!LockCacheItem::isLocked(FileName) &&
          isCacheItemSrcEqual(FileName + ".src", Devices[DeviceIndex],
                              SortedImgs, SpecConsts, BuildOptionsString)) {
        try {
          std::string FullFileName = FileName + ".bin";
          Binaries[DeviceIndex] = readBinaryDataFromFile(FullFileName);

          // Explicitly update the access time of the file. This is required for
          // eviction.
          if (isEvictionEnabled())
            saveCurrentTimeInAFile(FileName + CacheEntryAccessTimeSuffix);

          FileNames += FullFileName + ";";
          break;
        } catch (...) {
          // If read was unsuccessfull try the next item
        }
      }
      FileName = Path + "/" + std::to_string(++i);
    }
    // If there is no binary for any device, return empty vector.
    if (Binaries[DeviceIndex].empty())
      return {};
  }
  PersistentDeviceCodeCache::trace("using cached device binary: ", FileNames);
  return Binaries;
}

/*  kernel_compiler extension uses slightly different format for path
    and does not cache a .src separate from the binary.
 */
std::vector<std::vector<char>>
PersistentDeviceCodeCache::getCompiledKernelFromDisc(
    const std::vector<device> &Devices, const std::string &BuildOptionsString,
    const std::string &SourceStr) {
  assert(!Devices.empty());
  std::vector<std::vector<char>> Binaries(Devices.size());
  std::string FileNames;
  for (size_t DeviceIndex = 0; DeviceIndex < Devices.size(); DeviceIndex++) {
    std::string DirName = getCompiledKernelItemPath(
        Devices[DeviceIndex], BuildOptionsString, SourceStr);

    if (DirName.empty() || !OSUtil::isPathPresent(DirName))
      return {};

    int i = 0;
    std::string FileName{DirName + "/" + std::to_string(i)};
    while (OSUtil::isPathPresent(FileName + ".bin") ||
           OSUtil::isPathPresent(FileName + ".src")) {

      if (!LockCacheItem::isLocked(FileName)) {
        try {
          std::string FullFileName = FileName + ".bin";
          Binaries[DeviceIndex] = readBinaryDataFromFile(FullFileName);

          // Explicitly update the access time of the file. This is required for
          // eviction.
          if (isEvictionEnabled())
            saveCurrentTimeInAFile(FileName + CacheEntryAccessTimeSuffix);

          FileNames += FullFileName + ";";
          break;
        } catch (...) {
          // If read was unsuccessfull try the next item
        }
      }
      FileName = DirName + "/" + std::to_string(++i);
    }
    // If there is no binary for any device, return empty vector.
    if (Binaries[DeviceIndex].empty())
      return {};
  }
  PersistentDeviceCodeCache::trace_KernelCompiler("using cached binary: ",
                                                  FileNames);
  return Binaries;
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
 * Format: NumBinaries(=1), BinarySize, Binary
 */
void PersistentDeviceCodeCache::writeBinaryDataToFile(
    const std::string &FileName, const std::vector<char> &Data) {
  std::ofstream FileStream{FileName, std::ios::binary};
  // The reason why we need to write number of binaries (in current
  // implementation always 1) is to keep compatibility with the old format of
  // files in persistent cache, so that new runtime can use binaries from
  // persistent cache generated by old compiler/runtime.
  size_t NumBinaries = 1;
  FileStream.write((char *)&NumBinaries, sizeof(NumBinaries));

  auto Size = Data.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(Data.data(), Size);
  if (FileStream.fail())
    trace("Failed to write to binary file ", FileName);
}

/* Read built binary from persistent cache. Each persistent cache file contains
 * binary for a single device.
 * Format: NumBinaries(=1), BinarySize, Binary
 */
std::vector<char>
PersistentDeviceCodeCache::readBinaryDataFromFile(const std::string &FileName) {
  std::ifstream FileStream{FileName, std::ios::binary};
  // We ignore this number, we always read single device binary from a file and
  // we need this just to keep compatibility with the old format of files in
  // persistent cache, so that new runtime can use binaries from persistent
  // cache generated by old compiler/runtime.
  size_t NumBinaries = 0;
  FileStream.read((char *)&NumBinaries, sizeof(NumBinaries));
  if (FileStream.fail()) {
    trace("Failed to read number of binaries from ", FileName);
    return {};
  }
  // Even in the old implementation we could only put a single binary to the
  // persistent cache in all scenarios, multi-device case wasn't supported.
  assert(NumBinaries == 1);

  size_t BinarySize = 0;
  FileStream.read((char *)&BinarySize, sizeof(BinarySize));

  std::vector<char> BinaryData(BinarySize);
  FileStream.read(BinaryData.data(), BinarySize);
  FileStream.close();

  if (FileStream.fail()) {
    trace("Failed to read binary file from ", FileName);
    return {};
  }

  return BinaryData;
}

/* Writing cache item key sources to be used for reliable identification
 * Format: Four pairs of [size, value] for device, build options,
 * specialization constant values, device code SPIR-V images.
 */
void PersistentDeviceCodeCache::writeSourceItem(
    const std::string &FileName, const device &Device,
    const std::vector<const RTDeviceBinaryImage *> &SortedImgs,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  std::ofstream FileStream{FileName, std::ios::binary};

  std::string DeviceString{getDeviceIDString(Device)};
  size_t Size = DeviceString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(DeviceString.data(), Size);

  Size = BuildOptionsString.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write(BuildOptionsString.data(), Size);

  Size = SpecConsts.size();
  FileStream.write((char *)&Size, sizeof(Size));
  FileStream.write((const char *)SpecConsts.data(), Size);

  Size = 0;
  for (const RTDeviceBinaryImage *Img : SortedImgs)
    Size += Img->getSize();
  FileStream.write((char *)&Size, sizeof(Size));
  for (const RTDeviceBinaryImage *Img : SortedImgs)
    FileStream.write((const char *)Img->getRawData().BinaryStart,
                     Img->getSize());
  FileStream.close();

  if (FileStream.fail()) {
    trace("Failed to write source file to ", FileName);
  }
}

/* Check that cache item key sources are equal to the current program.
 * If file read operations fail cache item is treated as not equal.
 */
bool PersistentDeviceCodeCache::isCacheItemSrcEqual(
    const std::string &FileName, const device &Device,
    const std::vector<const RTDeviceBinaryImage *> &SortedImgs,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  std::ifstream FileStream{FileName, std::ios::binary};

  std::string ImgsString;
  for (const RTDeviceBinaryImage *Img : SortedImgs)
    ImgsString.append((const char *)Img->getRawData().BinaryStart,
                      Img->getSize());
  std::string SpecConstsString{(const char *)SpecConsts.data(),
                               SpecConsts.size()};

  size_t Size = 0;
  FileStream.read((char *)&Size, sizeof(Size));
  std::string res(Size, '\0');
  FileStream.read(&res[0], Size);
  if (getDeviceIDString(Device).compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (BuildOptionsString.compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (SpecConstsString.compare(res))
    return false;

  FileStream.read((char *)&Size, sizeof(Size));
  res.resize(Size);
  FileStream.read(&res[0], Size);
  if (ImgsString.compare(res))
    return false;

  FileStream.close();

  if (FileStream.fail()) {
    trace("Failed to read source file from ", FileName);
  }

  return true;
}

/* Returns directory name to store specific kernel images for specified
 * device, build options and specialization constants values.
 */
std::string PersistentDeviceCodeCache::getCacheItemPath(
    const device &Device, const std::vector<const RTDeviceBinaryImage *> &Imgs,
    const SerializedObj &SpecConsts, const std::string &BuildOptionsString) {
  std::string cache_root{getRootDir()};
  if (cache_root.empty()) {
    trace("Disable persistent cache due to unconfigured cache root.");
    return {};
  }

  std::string ImgsString;
  for (const RTDeviceBinaryImage *Img : Imgs)
    if (Img->getRawData().BinaryStart)
      ImgsString.append((const char *)Img->getRawData().BinaryStart,
                        Img->getSize());

  std::string DeviceString{getDeviceIDString(Device)};
  std::string SpecConstsString{(const char *)SpecConsts.data(),
                               SpecConsts.size()};
  std::hash<std::string> StringHasher{};

  return cache_root + "/" + std::to_string(StringHasher(DeviceString)) + "/" +
         std::to_string(StringHasher(ImgsString)) + "/" +
         std::to_string(StringHasher(SpecConstsString)) + "/" +
         std::to_string(StringHasher(BuildOptionsString));
}

std::string PersistentDeviceCodeCache::getCompiledKernelItemPath(
    const device &Device, const std::string &BuildOptionsString,
    const std::string &SourceString) {

  std::string cache_root{getRootDir()};
  if (cache_root.empty()) {
    trace("Disable persistent cache due to unconfigured cache root.");
    return {};
  }

  std::string DeviceString{getDeviceIDString(Device)};
  std::hash<std::string> StringHasher{};

  return cache_root + "/ext_kernel_compiler" + "/" +
         std::to_string(StringHasher(DeviceString)) + "/" +
         std::to_string(StringHasher(BuildOptionsString)) + "/" +
         std::to_string(StringHasher(SourceString));
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
} // namespace _V1
} // namespace sycl

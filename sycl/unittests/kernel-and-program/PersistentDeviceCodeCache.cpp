//==----- PersistentDeviceCodeCache.cpp --- Persistent cache tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains tests covering persistena device code cache functionality.
// Detailed description of the tests cases can be seen per test function.
#include "../thread_safety/ThreadUtils.h"
#include "detail/persistent_device_code_cache.hpp"
#include <detail/device_binary_image.hpp>
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <llvm/Support/FileSystem.h>
#include <sycl/detail/os_util.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <fstream>
#include <optional>
#include <vector>

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code EC = x) {                                                \
    FAIL() << #x ": did not return errc::success.\n"                           \
           << "error number: " << EC.value() << "\n"                           \
           << "error message: " << EC.message() << "\n";                       \
  }

// TODO: Introduce common unit tests header and move it there
static void set_env(const char *name, const char *value) {
#ifdef _WIN32
  (void)_putenv_s(name, value ? value : "");
#else
  if (value)
    (void)setenv(name, value, /*overwrite*/ 1);
  else
    (void)unsetenv(name);
#endif
}

namespace {
using namespace sycl;

/* Vector of programs which can be used for testing
 */
std::vector<std::vector<int>> Progs = {
    {128},   /*tiny program for 1 target device, 128 B long*/
    {10240}, /*small program for 1 target device, 10 kB long*/
    {1024 * 1024, 1024, 256, 1024 * 64}, /*big program for 4 target
                                           device, ~1 MB long*/
};

static unsigned char DeviceCodeID = 2;

static ur_result_t redefinedProgramGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(*params.ppPropValue);
    *value = Progs[DeviceCodeID].size();
  }

  if (*params.ppropName == UR_PROGRAM_INFO_DEVICES) {
    if (*params.ppPropValue) {
      for (size_t i = 0; i < Progs[DeviceCodeID].size(); i++) {
        auto devs = static_cast<ur_device_handle_t *>(*params.ppPropValue);
        devs[i] = reinterpret_cast<ur_device_handle_t>(i + 1);
      }
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet =
          sizeof(ur_device_handle_t) * Progs[DeviceCodeID].size();
    return UR_RESULT_SUCCESS;
  }

  if (*params.ppropName == UR_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(*params.ppPropValue);
    for (size_t i = 0; i < Progs[DeviceCodeID].size(); ++i)
      value[i] = Progs[DeviceCodeID][i];
  }

  if (*params.ppropName == UR_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char **>(*params.ppPropValue);
    for (size_t i = 0; i < Progs[DeviceCodeID].size(); ++i) {
      for (int j = 0; j < Progs[DeviceCodeID][i]; ++j) {
        value[i][j] = i;
      }
    }
  }

  return UR_RESULT_SUCCESS;
}

class PersistentDeviceCodeCache
    : public ::testing::TestWithParam<sycl_device_binary_type> {
public:
#ifdef _WIN32
  int setenv(const char *name, const char *value, int overwrite) {
    int errcode = 0;
    if (!overwrite) {
      size_t envsize = 0;
      errcode = getenv_s(&envsize, NULL, 0, name);
      if (errcode || envsize)
        return errcode;
    }
    return _putenv_s(name, value);
  }
#endif

  std::optional<std::string> SYCLCachePersistentBefore;
  bool SYCLCachePersistentChanged = false;

  std::string RootSYCLCacheDir;

  // Caches the initial value of the SYCL_CACHE_PERSISTENT environment variable
  // before overwriting it with the new value.
  // Tear-down will reset the environment variable.
  void SetSYCLCachePersistentEnv(const char *NewValue) {
    char *SYCLCachePersistent = getenv("SYCL_CACHE_PERSISTENT");
    // We can skip if the new value is the same as the old one.
    if ((!NewValue && !SYCLCachePersistent) ||
        (NewValue && SYCLCachePersistent &&
         !strcmp(NewValue, SYCLCachePersistent)))
      return;

    // Cache the old value of SYCL_CACHE_PERSISTENT if it is not already saved.
    if (!SYCLCachePersistentChanged && SYCLCachePersistent)
      SYCLCachePersistentBefore = std::string{SYCLCachePersistent};

    // Set the environment variable and signal the configuration file and the
    // persistent cache.
    set_env("SYCL_CACHE_PERSISTENT", NewValue);
    sycl::detail::SYCLConfig<sycl::detail::SYCL_CACHE_PERSISTENT>::reset();
    SYCLCachePersistentChanged = true;
  }

  // Set SYCL_CACHE_MAX_SIZE.
  void SetDiskCacheEvictionEnv(const char *NewValue) {
    set_env("SYCL_CACHE_MAX_SIZE", NewValue);
    sycl::detail::SYCLConfig<sycl::detail::SYCL_CACHE_MAX_SIZE>::reset();
  }

  void AppendToSYCLCacheDirEnv(const char *SubDir) {
    std::string NewSYCLCacheDirPath{RootSYCLCacheDir};
    if (NewSYCLCacheDirPath.back() != '\\' && NewSYCLCacheDirPath.back() != '/')
      NewSYCLCacheDirPath += '/';
    NewSYCLCacheDirPath += SubDir;
    set_env("SYCL_CACHE_DIR", NewSYCLCacheDirPath.c_str());
    sycl::detail::SYCLConfig<sycl::detail::SYCL_CACHE_DIR>::reset();
  }

  // Get the list of binary files in the cache directory.
  std::vector<std::string> getBinaryFileNames(std::string CachePath) {

    std::vector<std::string> FileNames;
    std::error_code EC;
    for (llvm::sys::fs::directory_iterator DirIt(CachePath, EC);
         DirIt != llvm::sys::fs::directory_iterator(); DirIt.increment(EC)) {
      // Check if the file is a binary file.
      std::string filename = DirIt->path();
      if (filename.find(".bin") != std::string::npos) {
        // Just return the file name without the path.
        FileNames.push_back(filename.substr(filename.find_last_of("/\\") + 1));
      }
    }

    return FileNames;
  }

  void ResetSYCLCacheDirEnv() {
    set_env("SYCL_CACHE_DIR", RootSYCLCacheDir.c_str());
    sycl::detail::SYCLConfig<sycl::detail::SYCL_CACHE_DIR>::reset();
  }

  void SetUp() override {
    if (RootSYCLCacheDir == "")
      FAIL() << "Please set SYCL_CACHE_DIR environment variable pointing to "
                "cache location.";

    // Append the test name to the cache dir to prevent conflicts with other
    // tests running in parallel.
    AppendToSYCLCacheDirEnv(
        ::testing::UnitTest::GetInstance()->current_test_info()->name());

    // Enable persistent cache
    SetSYCLCachePersistentEnv("1");
  }

  void TearDown() override {
    // If we changed the cache, set it back to the old value.
    if (SYCLCachePersistentChanged)
      SetSYCLCachePersistentEnv(SYCLCachePersistentBefore
                                    ? SYCLCachePersistentBefore->c_str()
                                    : nullptr);

    // Reset SYCL_CACHE_MAX_SIZE.
    SetDiskCacheEvictionEnv(nullptr);
    ResetSYCLCacheDirEnv();
  }

  PersistentDeviceCodeCache() : Mock{}, Plt{sycl::platform()} {

    char *SYCLCacheDir = getenv("SYCL_CACHE_DIR");
    if (!SYCLCacheDir) {
      std::clog << "This test requires the SYCL_CACHE_DIR environment variable "
                   "to be set.";
      return;
    }
    RootSYCLCacheDir = SYCLCacheDir;

    Dev = Plt.get_devices()[0];
    mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                            &redefinedProgramGetInfoAfter);
  }

  /* Helper function for concurent cache item read/write from different number
   * of threads with different cache item sizes:
   *  ProgramID   - defines program parameters to be used for testing (see Progs
   *              vector above.
   *  ThreadCount - number of parallel executors used for the test*/
  void ConcurentReadWriteCache(unsigned char ProgramID, size_t ThreadCount) {
    std::string BuildOptions{"--concurrent-access=" +
                             std::to_string(ThreadCount)};
    DeviceCodeID = ProgramID;
    std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
        {Dev}, {&Img}, {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID},
        BuildOptions);
    ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

    Barrier b(ThreadCount);
    {
      auto testLambda = [&](std::size_t threadId) {
        b.wait();
        detail::PersistentDeviceCodeCache::putItemToDisc(
            {Dev}, {&Img},
            std::vector<unsigned char>(
                {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID}),
            BuildOptions, NativeProg);
        auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
            {Dev}, {&Img},
            std::vector<unsigned char>(
                {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID}),
            BuildOptions);
        for (size_t i = 0; i < Res.size(); ++i) {
          for (size_t j = 0; j < Res[i].size(); ++j) {
            EXPECT_EQ(Res[i][j], static_cast<char>(i))
                << "Corrupted image loaded from persistent cache";
          }
        }
      };

      ThreadPool MPool(ThreadCount, testLambda);
    }
    ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
  }

protected:
  unittest::UrMock<> Mock;
  platform Plt;
  device Dev;
  const char *EntryName = "Entry";
  _sycl_offload_entry_struct EntryStruct = {
      /*addr*/ nullptr, const_cast<char *>(EntryName), strlen(EntryName),
      /*flags*/ 0, /*reserved*/ 0};
  sycl_device_binary_struct BinStruct{/*Version*/ 1,
                                      /*Kind*/ 4,
                                      /*Format*/ GetParam(),
                                      /*DeviceTargetSpec*/ nullptr,
                                      /*CompileOptions*/ nullptr,
                                      /*LinkOptions*/ nullptr,
                                      /*ManifestStart*/ nullptr,
                                      /*ManifestEnd*/ nullptr,
                                      /*BinaryStart*/ nullptr,
                                      /*BinaryEnd*/ nullptr,
                                      /*EntriesBegin*/ &EntryStruct,
                                      /*EntriesEnd*/ &EntryStruct + 1,
                                      /*PropertySetsBegin*/ nullptr,
                                      /*PropertySetsEnd*/ nullptr};
  sycl_device_binary Bin = &BinStruct;
  detail::RTDeviceBinaryImage Img{Bin};
  ur_program_handle_t NativeProg;
};

/* Checks that key values with \0 symbols are processed correctly
 */
TEST_P(PersistentDeviceCodeCache, KeysWithNullTermSymbol) {
  std::string Key{'1', '\0', '3', '4', '\0'};
  std::vector<unsigned char> SpecConst(Key.begin(), Key.end());
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, {&Img}, SpecConst, Key);
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, SpecConst,
                                                   Key, NativeProg);
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img},
                                                                SpecConst, Key);
  EXPECT_NE(Res.size(), static_cast<size_t>(0)) << "Failed to load cache item";
  for (size_t i = 0; i < Res.size(); ++i) {
    EXPECT_NE(Res[i].size(), static_cast<size_t>(0))
        << "Failed to load device image";
    for (size_t j = 0; j < Res[i].size(); ++j) {
      EXPECT_EQ(Res[i][j], static_cast<unsigned char>(i))
          << "Corrupted image loaded from persistent cache";
    }
  }

  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
}

TEST_P(PersistentDeviceCodeCache, MultipleImages) {
  const char *ExtraEntryName = "ExtraEntry";
  _sycl_offload_entry_struct ExtraEntryStruct = {
      /*addr*/ nullptr, const_cast<char *>(ExtraEntryName),
      strlen(ExtraEntryName), /*flags*/ 0, /*reserved*/ 0};
  sycl_device_binary_struct ExtraBinStruct{/*Version*/ 1,
                                           /*Kind*/ 4,
                                           /*Format*/ GetParam(),
                                           /*DeviceTargetSpec*/ nullptr,
                                           /*CompileOptions*/ nullptr,
                                           /*LinkOptions*/ nullptr,
                                           /*ManifestStart*/ nullptr,
                                           /*ManifestEnd*/ nullptr,
                                           /*BinaryStart*/ nullptr,
                                           /*BinaryEnd*/ nullptr,
                                           /*EntriesBegin*/ &ExtraEntryStruct,
                                           /*EntriesEnd*/ &ExtraEntryStruct + 1,
                                           /*PropertySetsBegin*/ nullptr,
                                           /*PropertySetsEnd*/ nullptr};
  sycl_device_binary ExtraBin = &ExtraBinStruct;
  detail::RTDeviceBinaryImage ExtraImg{ExtraBin};
  std::string BuildOptions{"--multiple-images"};

  std::vector<const detail::RTDeviceBinaryImage *> Imgs{&Img, &ExtraImg};
  // Images are supposed to be sorted before requesting cache item path.
  std::sort(Imgs.begin(), Imgs.end(),
            [](const detail::RTDeviceBinaryImage *A,
               const detail::RTDeviceBinaryImage *B) {
              return std::strcmp(A->getRawData().EntriesBegin->name,
                                 B->getRawData().EntriesBegin->name) < 0;
            });
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, Imgs, {}, BuildOptions);
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, Imgs, {},
                                                   BuildOptions, NativeProg);
  // Check that the order of images does not affect the result.
  std::reverse(Imgs.begin(), Imgs.end());
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, Imgs, {},
                                                                BuildOptions);
  EXPECT_NE(Res.size(), static_cast<size_t>(0)) << "Failed to load cache item";
  for (size_t i = 0; i < Res.size(); ++i) {
    EXPECT_NE(Res[i].size(), static_cast<size_t>(0))
        << "Failed to load device image";
    for (size_t j = 0; j < Res[i].size(); ++j) {
      EXPECT_EQ(Res[i][j], static_cast<unsigned char>(i))
          << "Corrupted image loaded from persistent cache";
    }
  }

  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
}

/* Do read/write for the same cache item to/from 300 threads for small device
 * code size. Make sure that there is no data corruption or crashes.
 */
TEST_P(PersistentDeviceCodeCache, ConcurentReadWriteSmallItem) {
  ConcurentReadWriteCache(0, 300);
}

/* Do read/write for the same cache item to/from 100 threads for medium device
 * code size. Make sure that there is no data corruption or crashes.
 */
TEST_P(PersistentDeviceCodeCache, ConcurentReadWriteCacheMediumItem) {
  ConcurentReadWriteCache(1, 100);
}

/* Do read/write for the same cache item to/from 20 threads from big device
 * code size. Make sure that there is no data corruption or crashes.
 */
TEST_P(PersistentDeviceCodeCache, ConcurentReadWriteCacheBigItem) {
  ConcurentReadWriteCache(2, 20);
}

/* Checks that no crash happens when cache items are corrupted on cache read.
 * The case when source or binary files are corrupted is treated as cache miss.
 *  - only source file is present;
 *  - only binary file is present;
 *  - source file is corrupted;
 *  - binary file is corrupted.
 */
TEST_P(PersistentDeviceCodeCache, CorruptedCacheFiles) {
  std::string BuildOptions{"--corrupted-file"};
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, {&Img}, {}, BuildOptions);
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Only source file is present
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_FALSE(llvm::sys::fs::remove(ItemDir + "/0.bin"))
      << "Failed to remove binary file";
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      {Dev}, {&Img}, {}, BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with missed binary file was read";
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Only binary file is present
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_FALSE(llvm::sys::fs::remove(ItemDir + "/0.src"))
      << "Failed to remove source file";
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img}, {},
                                                           BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with missed source file was read";
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Binary file is corrupted
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  std::ofstream FileStream(ItemDir + "/0.bin",
                           std::ofstream::out | std::ofstream::trunc);
  /* Emulate binary built for 2 devices: first is OK, second is trancated
   * from 23 bytes to 4
   */
  FileStream << 2 << 12 << "123456789012" << 23 << "1234";
  FileStream.close();
  EXPECT_FALSE(FileStream.fail()) << "Failed to create trancated binary file";
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img}, {},
                                                           BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with corrupted binary file was read";

  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Source file is empty
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  {
    std::ofstream FileStream(ItemDir + "/0.src",
                             std::ofstream::out | std::ofstream::trunc);
  }
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img}, {},
                                                           BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with corrupted binary file was read";
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
}

/* Checks that lock file affects cache operations as expected:
 *  - new cache item is created if existing one is locked on write operation;
 *  - cache miss happens on read operation.
 */
TEST_P(PersistentDeviceCodeCache, LockFile) {
  std::string BuildOptions{"--obsolete-lock"};
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, {&Img}, {}, BuildOptions);
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Create 1st cahe item
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/0.bin")) << "No file created";
  std::string LockFile = ItemDir + "/0.lock";
  EXPECT_FALSE(llvm::sys::fs::exists(LockFile)) << "Cache item locked";

  // Create lock file for the 1st cache item
  { std::ofstream File{LockFile}; }

  // Cache item is locked, cache miss happens on read
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      {Dev}, {&Img}, {}, BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0)) << "Locked item was read";

  // Cache item is locked - new cache item to be created
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/1.bin")) << "No file created";

  // Second cache item is locked, cache miss happens on read
  { std::ofstream File{ItemDir + "/1.lock"}; }
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img}, {},
                                                           BuildOptions);

  EXPECT_EQ(Res.size(), static_cast<size_t>(0)) << "Locked item was read";

  // First cache item was unlocked and successfully read
  std::remove(LockFile.c_str());
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img}, {},
                                                           BuildOptions);
  for (size_t i = 0; i < Res.size(); ++i) {
    for (size_t j = 0; j < Res[i].size(); ++j) {
      EXPECT_EQ(Res[i][j], static_cast<unsigned char>(i))
          << "Corrupted image loaded from persistent cache";
    }
  }
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
}

#ifndef _WIN32
// llvm::sys::fs::setPermissions does not make effect on Windows
/* Checks cache behavior when filesystem read/write operations fail
 */
TEST_P(PersistentDeviceCodeCache, AccessDeniedForCacheDir) {
  std::string BuildOptions{"--build-options"};
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, {&Img}, {}, BuildOptions);
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/0.bin")) << "No file created";
  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/0.bin",
                                                llvm::sys::fs::no_perms));
  // No access to binary file new cache item to be created
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/1.bin")) << "No file created";

  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/1.bin",
                                                llvm::sys::fs::no_perms));
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      {Dev}, {&Img}, {}, BuildOptions);

  // No image to be read due to lack of permissions from source file
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Read from the file without permissions.";

  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/0.bin",
                                                llvm::sys::fs::all_perms));
  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/1.bin",
                                                llvm::sys::fs::all_perms));

  Res = detail::PersistentDeviceCodeCache::getItemFromDisc({Dev}, {&Img}, {},
                                                           BuildOptions);
  // Image should be successfully read
  for (size_t i = 0; i < Res.size(); ++i) {
    for (size_t j = 0; j < Res[i].size(); ++j) {
      EXPECT_EQ(Res[i][j], static_cast<unsigned char>(i))
          << "Corrupted image loaded from persistent cache";
    }
  }
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
}
#endif //_WIN32

// Unit tests for testing eviction in persistent cache.
TEST_P(PersistentDeviceCodeCache, BasicEviction) {

  // Cleanup the cache directory.
  std::string CacheRoot = detail::PersistentDeviceCodeCache::getRootDir();
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(CacheRoot));
  ASSERT_NO_ERROR(llvm::sys::fs::create_directories(CacheRoot));

  // Disable eviction for the time being.
  SetDiskCacheEvictionEnv("9000000");

  std::string BuildOptions{"--eviction"};
  // Put 3 items to the cache.
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);

  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);

  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);

  // Retrieve 0.bin from the cache.
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      {Dev}, {&Img}, {}, BuildOptions);

  // Get the number of binary files in the cached item folder.
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, {&Img}, {}, BuildOptions);
  auto BinFiles = getBinaryFileNames(ItemDir);

  EXPECT_EQ(BinFiles.size(), static_cast<size_t>(3))
      << "Missing binary files. Eviction should not have happened.";

  // Get Cache size and size of each entry. Set eviction threshold so that
  // just one item is evicted.
  size_t SizeOfOneEntry =
      (size_t)(detail::getDirectorySize(CacheRoot, false)) + 10;

  // Set SYCL_CACHE_MAX_SIZE.
  SetDiskCacheEvictionEnv(std::to_string(SizeOfOneEntry).c_str());

  // Put 4th item to the cache. This should trigger eviction. Only the first
  // item should be evicted.
  detail::PersistentDeviceCodeCache::putItemToDisc({Dev}, {&Img}, {},
                                                   BuildOptions, NativeProg);

  // We should have three binary files: 0.bin, 2.bin, 3.bin.
  BinFiles = getBinaryFileNames(ItemDir);
  EXPECT_EQ(BinFiles.size(), static_cast<size_t>(3))
      << "Eviction failed. Wrong number of binary files in the cache.";

  // Check that 1.bin was evicted.
  for (const auto &File : BinFiles)
    EXPECT_NE(File, "1.bin")
        << "Eviction failed. 1.bin should have been evicted.";

  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));
}

// Unit test for testing size file creation and update, concurrently.
TEST_P(PersistentDeviceCodeCache, ConcurentReadWriteCacheFileSize) {
  // Cleanup the cache directory.
  std::string CacheRoot = detail::PersistentDeviceCodeCache::getRootDir();
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(CacheRoot));
  ASSERT_NO_ERROR(llvm::sys::fs::create_directories(CacheRoot));

  // Insanely large value (1GB) to not trigger eviction. This test just checks
  // for deadlocks/crashes when updating the size file concurrently.
  SetDiskCacheEvictionEnv("1000000000");
  ConcurentReadWriteCache(1, 50);
}

// Unit test for adding and evicting cache, concurrently.
TEST_P(PersistentDeviceCodeCache, ConcurentReadWriteCacheEviction) {
  // Cleanup the cache directory.
  std::string CacheRoot = detail::PersistentDeviceCodeCache::getRootDir();
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(CacheRoot));
  ASSERT_NO_ERROR(llvm::sys::fs::create_directories(CacheRoot));

  SetDiskCacheEvictionEnv("1000");
  ConcurentReadWriteCache(2, 40);
}

INSTANTIATE_TEST_SUITE_P(PersistentDeviceCodeCacheImpl,
                         PersistentDeviceCodeCache,
                         ::testing::Values(SYCL_DEVICE_BINARY_TYPE_SPIRV,
                                           SYCL_DEVICE_BINARY_TYPE_NATIVE));
} // namespace

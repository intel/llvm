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
#include <helpers/PiMock.hpp>
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

static pi_result redefinedProgramGetInfoAfter(pi_program program,
                                              pi_program_info param_name,
                                              size_t param_value_size,
                                              void *param_value,
                                              size_t *param_value_size_ret) {
  if (param_name == PI_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(param_value);
    *value = Progs[DeviceCodeID].size();
  }

  if (param_name == PI_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(param_value);
    for (size_t i = 0; i < Progs[DeviceCodeID].size(); ++i)
      value[i] = Progs[DeviceCodeID][i];
  }

  if (param_name == PI_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char **>(param_value);
    for (size_t i = 0; i < Progs[DeviceCodeID].size(); ++i) {
      for (int j = 0; j < Progs[DeviceCodeID][i]; ++j) {
        value[i][j] = i;
      }
    }
  }

  return PI_SUCCESS;
}

class PersistentDeviceCodeCache
    : public ::testing::TestWithParam<pi_device_binary_type> {
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

  void AppendToSYCLCacheDirEnv(const char *SubDir) {
    std::string NewSYCLCacheDirPath{RootSYCLCacheDir};
    if (NewSYCLCacheDirPath.back() != '\\' && NewSYCLCacheDirPath.back() != '/')
      NewSYCLCacheDirPath += '/';
    NewSYCLCacheDirPath += SubDir;
    set_env("SYCL_CACHE_DIR", NewSYCLCacheDirPath.c_str());
    sycl::detail::SYCLConfig<sycl::detail::SYCL_CACHE_DIR>::reset();
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
    ResetSYCLCacheDirEnv();
  }

  PersistentDeviceCodeCache() : Mock{}, Plt{Mock.getPlatform()} {

    char *SYCLCacheDir = getenv("SYCL_CACHE_DIR");
    if (!SYCLCacheDir) {
      std::clog << "This test requires the SYCL_CACHE_DIR environment variable "
                   "to be set.";
      return;
    }
    RootSYCLCacheDir = SYCLCacheDir;

    Dev = Plt.get_devices()[0];
    Mock.redefineAfter<detail::PiApiKind::piProgramGetInfo>(
        redefinedProgramGetInfoAfter);
  }

  /* Helper function for concurent cache item read/write from diffrent number
   * of threads with diffrent cache item sizes:
   *  ProgramID   - defines program parameters to be used for testing (see Progs
   *              vector above.
   *  ThreadCount - number of parallel executors used for the test*/
  void ConcurentReadWriteCache(unsigned char ProgramID, size_t ThreadCount) {
    std::string BuildOptions{"--concurrent-access=" +
                             std::to_string(ThreadCount)};
    DeviceCodeID = ProgramID;
    std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
        Dev, {&Img}, {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID},
        BuildOptions);
    ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

    Barrier b(ThreadCount);
    {
      auto testLambda = [&](std::size_t threadId) {
        b.wait();
        detail::PersistentDeviceCodeCache::putItemToDisc(
            Dev, {&Img},
            std::vector<unsigned char>(
                {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID}),
            BuildOptions, NativeProg);
        auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(
            Dev, {&Img},
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
  unittest::PiMock Mock;
  platform Plt;
  device Dev;
  const char *EntryName = "Entry";
  _pi_offload_entry_struct EntryStruct = {
      /*addr*/ nullptr, const_cast<char *>(EntryName), strlen(EntryName),
      /*flags*/ 0, /*reserved*/ 0};
  pi_device_binary_struct BinStruct{/*Version*/ 1,
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
  pi_device_binary Bin = &BinStruct;
  detail::RTDeviceBinaryImage Img{Bin};
  sycl::detail::pi::PiProgram NativeProg;
};

/* Checks that key values with \0 symbols are processed correctly
 */
TEST_P(PersistentDeviceCodeCache, KeysWithNullTermSymbol) {
  std::string Key{'1', '\0', '3', '4', '\0'};
  std::vector<unsigned char> SpecConst(Key.begin(), Key.end());
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, {&Img}, SpecConst, Key);
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, SpecConst, Key,
                                                   NativeProg);
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img},
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
  _pi_offload_entry_struct ExtraEntryStruct = {
      /*addr*/ nullptr, const_cast<char *>(ExtraEntryName),
      strlen(ExtraEntryName), /*flags*/ 0, /*reserved*/ 0};
  pi_device_binary_struct ExtraBinStruct{/*Version*/ 1,
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
  pi_device_binary ExtraBin = &ExtraBinStruct;
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

  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, Imgs, {}, BuildOptions,
                                                   NativeProg);
  // Check that the order of images does not affect the result.
  std::reverse(Imgs.begin(), Imgs.end());
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, Imgs, {},
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
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_FALSE(llvm::sys::fs::remove(ItemDir + "/0.bin"))
      << "Failed to remove binary file";
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
                                                                BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with missed binary file was read";
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Only binary file is present
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_FALSE(llvm::sys::fs::remove(ItemDir + "/0.src"))
      << "Failed to remove source file";
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
                                                           BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with missed source file was read";
  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Binary file is corrupted
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  std::ofstream FileStream(ItemDir + "/0.bin",
                           std::ofstream::out | std::ofstream::trunc);
  /* Emulate binary built for 2 devices: first is OK, second is trancated
   * from 23 bytes to 4
   */
  FileStream << 2 << 12 << "123456789012" << 23 << "1234";
  FileStream.close();
  EXPECT_FALSE(FileStream.fail()) << "Failed to create trancated binary file";
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
                                                           BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Item with corrupted binary file was read";

  ASSERT_NO_ERROR(llvm::sys::fs::remove_directories(ItemDir));

  // Source file is empty
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  {
    std::ofstream FileStream(ItemDir + "/0.src",
                             std::ofstream::out | std::ofstream::trunc);
  }
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
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
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/0.bin")) << "No file created";
  std::string LockFile = ItemDir + "/0.lock";
  EXPECT_FALSE(llvm::sys::fs::exists(LockFile)) << "Cache item locked";

  // Create lock file for the 1st cache item
  { std::ofstream File{LockFile}; }

  // Cache item is locked, cache miss happens on read
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
                                                                BuildOptions);
  EXPECT_EQ(Res.size(), static_cast<size_t>(0)) << "Locked item was read";

  // Cache item is locked - new cache item to be created
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/1.bin")) << "No file created";

  // Second cache item is locked, cache miss happens on read
  { std::ofstream File{ItemDir + "/1.lock"}; }
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
                                                           BuildOptions);

  EXPECT_EQ(Res.size(), static_cast<size_t>(0)) << "Locked item was read";

  // First cache item was unlocked and successfully read
  std::remove(LockFile.c_str());
  Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
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
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/0.bin")) << "No file created";
  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/0.bin",
                                                llvm::sys::fs::no_perms));
  // No access to binary file new cache item to be created
  detail::PersistentDeviceCodeCache::putItemToDisc(Dev, {&Img}, {},
                                                   BuildOptions, NativeProg);
  EXPECT_TRUE(llvm::sys::fs::exists(ItemDir + "/1.bin")) << "No file created";

  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/1.bin",
                                                llvm::sys::fs::no_perms));
  auto Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
                                                                BuildOptions);

  // No image to be read due to lack of permissions from source file
  EXPECT_EQ(Res.size(), static_cast<size_t>(0))
      << "Read from the file without permissions.";

  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/0.bin",
                                                llvm::sys::fs::all_perms));
  ASSERT_NO_ERROR(llvm::sys::fs::setPermissions(ItemDir + "/1.bin",
                                                llvm::sys::fs::all_perms));

  Res = detail::PersistentDeviceCodeCache::getItemFromDisc(Dev, {&Img}, {},
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

INSTANTIATE_TEST_SUITE_P(PersistentDeviceCodeCacheImpl,
                         PersistentDeviceCodeCache,
                         ::testing::Values(PI_DEVICE_BINARY_TYPE_SPIRV,
                                           PI_DEVICE_BINARY_TYPE_NATIVE));
} // namespace

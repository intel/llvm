//==----- PersistenDeviceCodeCache.cpp --- Persistent cache tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../thread_safety/ThreadUtils.h"
#include "detail/persistent_device_code_cache.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/device_binary_image.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <llvm/Support/FileSystem.h>
#include <mutex>
#include <vector>

namespace {
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
using namespace cl::sycl;

/* Vector of programs which can be used for testing
 */
std::vector<std::vector<int>> Progs = {
    {128},   /*tiny program for 1 target device, 128 B long*/
    {10240}, /*small program for 1 target device, 10 kB long*/
    {1024 * 1024, 1024, 256, 1024 * 64}, /*big program for 4 target
                                           device, ~1 MB long*/
};

static unsigned char DeviceCodeID = 2;

static pi_result redefinedProgramGetInfo(pi_program program,
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
    for (int i = 0; i < Progs[DeviceCodeID].size(); ++i)
      value[i] = Progs[DeviceCodeID][i];
  }

  if (param_name == PI_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char **>(param_value);
    for (int i = 0; i < Progs[DeviceCodeID].size(); ++i)
      for (int j = 0; j < Progs[DeviceCodeID][i]; ++j)
        value[i][j] = i;
  }

  return PI_SUCCESS;
}

class PersistenDeviceCodeCache : public ::testing::Test {
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

  PersistenDeviceCodeCache() : Plt{default_selector()} {

    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      std::clog << "This test is only supported on OpenCL devices\n";
      std::clog << "Current platform is "
                << Plt.get_info<info::platform::name>();
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);
    Dev = Plt.get_devices()[0];
    Mock->redefine<detail::PiApiKind::piProgramGetInfo>(
        redefinedProgramGetInfo);
  }

  void ConcurentReadWriteCache(unsigned char ProgramID, size_t ThreadCount) {
    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      return;
    }

    DeviceCodeID = ProgramID;
    std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
        Dev, Img, {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID},
        "--build-options");

    Barrier b(ThreadCount);
    {
      auto testLambda = [&](std::size_t threadId) {
        b.wait();
        detail::PersistentDeviceCodeCache::putItemToDisc(
            Dev, Img,
            sycl::vector_class<unsigned char>(
                {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID}),
            "--build-options", NativeProg);
        auto res = detail::PersistentDeviceCodeCache::getItemFromDisc(
            Dev, Img,
            sycl::vector_class<unsigned char>(
                {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't', ProgramID}),
            "--build-options", NativeProg);
        for (int i = 0; i < res.size(); ++i) {
          for (int j = 0; j < res[i].size(); ++j) {
            assert(res[i][j] == i &&
                   "Corrupted image loaded from persistent cache");
          }
        }
      };

      ThreadPool MPool(ThreadCount, testLambda);
    }
    llvm::sys::fs::remove_directories(ItemDir);
  }

protected:
  detail::OSModuleHandle ModuleHandle = detail::OSUtil::ExeModuleHandle;
  platform Plt;
  device Dev;
  pi_device_binary_struct BinStruct{/*Version*/ 1, /*Kind*/ 4,
                                    /*Format*/ PI_DEVICE_BINARY_TYPE_SPIRV};
  pi_device_binary Bin = &BinStruct;
  detail::RTDeviceBinaryImage Img{Bin, ModuleHandle};
  RT::PiProgram NativeProg;
  std::unique_ptr<unittest::PiMock> Mock;
};

/* Do read/write for the same cache item to/from 300 threads for small device
 * code size. Make sure that there is no data corruption or crashes.
 */
TEST_F(PersistenDeviceCodeCache, ConcurentReadWriteSmallItem) {
  ConcurentReadWriteCache(0, 300);
}

/* Do read/write for the same cache item to/from 100 threads for medium device
 * code size. Make sure that there is no data corruption or crashes.
 */
TEST_F(PersistenDeviceCodeCache, ConcurentReadWriteCacheMediumItem) {
  ConcurentReadWriteCache(1, 100);
}

/* Do read/write for the same cache item to/from 20 threads from big device
 * code size. Make sure that there is no data corruption or crashes.
 */
TEST_F(PersistenDeviceCodeCache, ConcurentReadWriteCacheBigItem) {
  ConcurentReadWriteCache(2, 20);
}

#ifndef _WIN32
// llvm::sys::fs::setPermissions doe not make effect on Windows
/* Checks cache behavior when filesystem read/write operations fail
 */
TEST_F(PersistenDeviceCodeCache, AccessDeniedForCacheDir) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }
  std::string ItemDir = detail::PersistentDeviceCodeCache::getCacheItemPath(
      Dev, Img, {}, "--build-options");
  detail::PersistentDeviceCodeCache::putItemToDisc(
      Dev, Img, {}, "--build-options", NativeProg);
  assert(llvm::sys::fs::exists(ItemDir + "/0.bin") && "No file created");
  llvm::sys::fs::setPermissions(ItemDir + "/0.bin", llvm::sys::fs::no_perms);
  // No access to binary file new cache item to be created
  detail::PersistentDeviceCodeCache::putItemToDisc(
      Dev, Img, {}, "--build-options", NativeProg);
  assert(llvm::sys::fs::exists(ItemDir + "/1.bin") && "No file created");

  llvm::sys::fs::setPermissions(ItemDir + "/1.bin", llvm::sys::fs::no_perms);
  auto res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      Dev, Img, {}, "--build-options", NativeProg);

  // No image to be read due to lack of permissions fro source file
  assert(res.size() == 0);

  llvm::sys::fs::setPermissions(ItemDir + "/0.bin", llvm::sys::fs::all_perms);
  llvm::sys::fs::setPermissions(ItemDir + "/1.bin", llvm::sys::fs::all_perms);

  res = detail::PersistentDeviceCodeCache::getItemFromDisc(
      Dev, Img, {}, "--build-options", NativeProg);
  // Image should be successfully read
  for (int i = 0; i < res.size(); ++i) {
    for (int j = 0; j < res[i].size(); ++j) {
      assert(res[i][j] == i && "Corrupted image loaded from persistent cache");
    }
  }
  llvm::sys::fs::remove_directories(ItemDir);
}
#endif //_WIN32
} // namespace

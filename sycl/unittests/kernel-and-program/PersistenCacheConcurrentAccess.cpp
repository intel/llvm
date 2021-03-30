//==----- PersistenCacheConcurrentAccess.cpp --- Persistent cache tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../thread_safety/ThreadUtils.h"
#include "detail/persistent_cache.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/device_binary_image.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <mutex>
#include <vector>

namespace {
constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
using namespace cl::sycl;
constexpr size_t BinNum = 4;
constexpr size_t BinSizes[BinNum] = {1024, 1024 * 1024, 256, 1024 * 64};
static pi_result redefinedProgramGetInfo(pi_program program,
                                         pi_program_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret) {
  if (param_name == PI_PROGRAM_INFO_NUM_DEVICES) {
    auto value = reinterpret_cast<unsigned int *>(param_value);
    *value = BinNum;
  }

  if (param_name == PI_PROGRAM_INFO_BINARY_SIZES) {
    auto value = reinterpret_cast<size_t *>(param_value);
    for (int i = 0; i < BinNum; ++i)
      value[i] = BinSizes[i];
  }

  if (param_name == PI_PROGRAM_INFO_BINARIES) {
    auto value = reinterpret_cast<unsigned char **>(param_value);
    for (int i = 0; i < BinNum; ++i)
      for (int j = 0; j < BinSizes[i]; ++j)
        value[i][j] = i;
  }

  return PI_SUCCESS;
}

class PersistenCacheConcurrentAccess : public ::testing::Test {
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

  PersistenCacheConcurrentAccess() : Plt{default_selector()} {
    const char *envTmp =
#ifdef _WIN32
        std::getenv("TEMP");
#else
        std::getenv("TMP");
#endif
    if (envTmp != nullptr)
      cacheRoot += envTmp;
    else
#ifdef _WIN32
      cacheRoot += "C:/temp";
#else
      cacheRoot += "/tmp";
#endif
    cacheRoot += "/PersistenCache";
    setenv("SYCL_CACHE_DIR", cacheRoot.c_str(), 0);
    std::printf("Use %s as cache root\n", cacheRoot.c_str());

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
    std::filesystem::remove_all(cacheRoot);
  }

protected:
  std::string cacheRoot;
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
TEST_F(PersistenCacheConcurrentAccess, ReadWriteCacheItem) {
  std::vector<std::vector<char>> Data = {
      std::vector<char>(1024, '1'), std::vector<char>(1024 * 1024, '2'),
      std::vector<char>(256, '3'), std::vector<char>(1024 * 64, '4')};

  constexpr std::size_t threadCount = 300;

  Barrier b(threadCount);
  {
    auto testLambda = [&](std::size_t threadId) {
      b.wait();
      detail::PersistentCache::putPIProgramToDisc(
          detail::getSyclObjImpl(Plt)->getPlugin(), Dev, Img,
          sycl::vector_class<unsigned char>(
              {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't'}),
          "--build-options", NativeProg);
      auto res = detail::PersistentCache::getPIProgramFromDisc(
          Dev, Img,
          sycl::vector_class<unsigned char>(
              {'S', 'p', 'e', 'c', 'C', 'o', 'n', 's', 't'}),
          "--build-options", NativeProg);
      for (int i = 0; i < res.size(); ++i) {
        for (int j = 0; j < res[i].size(); ++j) {
          assert(res[i][j] == i &&
                 "Corrupted image loaded from persistent cache");
        }
      }
    };

    ThreadPool MPool(threadCount, testLambda);
  }
}
} // namespace

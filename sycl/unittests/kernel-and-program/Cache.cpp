//==--------- Cache.cpp --- kernel and program cache unit test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include "CL/sycl/detail/pi.h"
#include "detail/context_impl.hpp"
#include "detail/kernel_program_cache.hpp"
#include "detail/program_impl.hpp"
#include <CL/sycl.hpp>
#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <iostream>

using namespace sycl;

class TestKernel {
public:
  void operator()(cl::sycl::item<1>){};
};

class TestKernel2 {
public:
  void operator()(cl::sycl::item<1>){};
};

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
struct MockKernelInfo {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

template <> struct KernelInfo<TestKernel> : public MockKernelInfo {
  static constexpr const char *getName() { return "TestKernel"; }
};

template <> struct KernelInfo<TestKernel2> : public MockKernelInfo {
  static constexpr const char *getName() { return "TestKernel2"; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({"TestKernel", "TestKernel2"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Img = generateDefaultImage();
static sycl::unittest::PiImageArray<1> ImgArray{&Img};

struct TestCtx {
  detail::pi::PiContext context;
};

std::unique_ptr<TestCtx> globalCtx;

static pi_result redefinedProgramCreateWithSource(pi_context context,
                                                  pi_uint32 count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

static pi_result redefinedProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t metadata_length, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(1);
  return PI_SUCCESS;
}

static pi_result redefinedKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_INFO_CONTEXT) {
    auto ctx = reinterpret_cast<detail::pi::PiContext *>(param_value);
    *ctx = globalCtx->context;
  }

  return PI_SUCCESS;
}

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  return PI_SUCCESS;
}
static pi_result redefinedKernelRelease(pi_kernel kernel) { return PI_SUCCESS; }

class KernelAndProgramCacheTest : public ::testing::Test {
public:
  KernelAndProgramCacheTest() : Plt{default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      std::clog << "This test is only supported on OpenCL devices\n";
      std::clog << "Current platform is "
                << Plt.get_info<info::platform::name>();
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);

    setupDefaultMockAPIs(*Mock);
    Mock->redefine<detail::PiApiKind::piclProgramCreateWithSource>(
        redefinedProgramCreateWithSource);
    Mock->redefine<detail::PiApiKind::piProgramCreateWithBinary>(
        redefinedProgramCreateWithBinary);
    Mock->redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);
    Mock->redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
    Mock->redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  }

protected:
  platform Plt;
  std::unique_ptr<unittest::PiMock> Mock;
};

// Check that programs built from source are not cached.
TEST_F(KernelAndProgramCacheTest, ProgramSourceNegativeBuild) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_source("");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs built from source with options are not cached.
TEST_F(KernelAndProgramCacheTest, ProgramSourceNegativeBuildWithOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_source("", "-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs compiled and linked from source are not cached.
TEST_F(KernelAndProgramCacheTest, ProgramSourceNegativeCompileAndLink) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_source("");
  Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs compiled and linked from source with options are not
// cached.
TEST_F(KernelAndProgramCacheTest, ProgramSourceNegativeCompileAndLinkWithOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_source("");
  Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs built without options are cached.
TEST_F(KernelAndProgramCacheTest, ProgramBuildPositive) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg1{Ctx};
  program Prg2{Ctx};

  Prg1.build_with_kernel_type<TestKernel>();
  Prg2.build_with_kernel_type<TestKernel>();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for programs";
}

// Check that programs built with options are cached.
TEST_F(KernelAndProgramCacheTest, ProgramBuildPositiveBuildOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg1{Ctx};
  program Prg2{Ctx};
  program Prg3{Ctx};
  program Prg4{Ctx};
  program Prg5{Ctx};

  /* Build 5 instances of the same program. It is expected that there will be 3
   * instances of the program in the cache because Build of Prg1 is equal to
   * build of Prg5 and build of Prg2 is equal to build of Prg3.
   * */
  Prg1.build_with_kernel_type<TestKernel>("-a");
  Prg2.build_with_kernel_type<TestKernel>("-b");
  Prg3.build_with_kernel_type<TestKernel>("-b");
  Prg4.build_with_kernel_type<TestKernel>();
  Prg5.build_with_kernel_type<TestKernel2>("-a");

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 3U) << "Expect non-empty cache for programs";
}

// Check that programs built with compile options are not cached.
TEST_F(KernelAndProgramCacheTest, ProgramBuildNegativeCompileOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>("-g");
  Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for programs";
}

// Check that programs built with link options are not cached.
TEST_F(KernelAndProgramCacheTest, ProgramBuildNegativeLinkOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>();
  Prg.link("-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for programs";
}

// Check that kernels built without options are cached.
TEST_F(KernelAndProgramCacheTest, KernelPositive) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>();
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with options are cached.
TEST_F(KernelAndProgramCacheTest, KernelPositiveBuildOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>("-g");

  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with compile options are not cached.
TEST_F(KernelAndProgramCacheTest, KernelNegativeCompileOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>("-g");
  Prg.link();
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels built with link options are not cached.
TEST_F(KernelAndProgramCacheTest, KernelNegativeLinkOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>();
  Prg.link("-g");
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels are not cached if program is created from multiple
// programs.
TEST_F(KernelAndProgramCacheTest, KernelNegativeLinkedProgs) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg1{Ctx};
  program Prg2{Ctx};

  Prg1.compile_with_kernel_type<TestKernel>();
  Prg2.compile_with_kernel_type<TestKernel2>();
  program Prg({Prg1, Prg2});
  kernel Ker = Prg.get_kernel<TestKernel>();

  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels created from source are not cached.
TEST_F(KernelAndProgramCacheTest, KernelNegativeSource) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_source("");
  kernel Ker = Prg.get_kernel("test");

  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

typedef KernelAndProgramCacheTest KernelAndProgramFastCacheTest;

class MockKernelProgramCache : public detail::KernelProgramCache {
public:
  static detail::KernelProgramCache::KernelFastCacheT &
  getFastCache(detail::KernelProgramCache &cache) {
    return (reinterpret_cast<MockKernelProgramCache &>(cache)).get();
  }

  detail::KernelProgramCache::KernelFastCacheT &get() {
    return this->MKernelFastCache;
  }
};

// Check that kernels built without options are cached.
TEST_F(KernelAndProgramFastCacheTest, KernelPositive) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>();
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with options are cached.
TEST_F(KernelAndProgramFastCacheTest, KernelPositiveBuildOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>("-g");

  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with compile options are not cached.
TEST_F(KernelAndProgramFastCacheTest, KernelNegativeCompileOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>("-g");
  Prg.link();
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels built with link options are not cached.
TEST_F(KernelAndProgramFastCacheTest, KernelNegativeLinkOpts) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>();
  Prg.link("-g");
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels are not cached if program is created from multiple
// programs.
TEST_F(KernelAndProgramFastCacheTest, KernelNegativeLinkedProgs) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg1{Ctx};
  program Prg2{Ctx};

  Prg1.compile_with_kernel_type<TestKernel>();
  Prg2.compile_with_kernel_type<TestKernel2>();
  program Prg({Prg1, Prg2});
  kernel Ker = Prg.get_kernel<TestKernel>();

  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels created from source are not cached.
TEST_F(KernelAndProgramFastCacheTest, KernelNegativeSource) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_source("");
  kernel Ker = Prg.get_kernel("test");

  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

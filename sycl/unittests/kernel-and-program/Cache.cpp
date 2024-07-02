//==--------- Cache.cpp --- kernel and program cache unit test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// All these tests are temporarily disabled, since they need to be rewrited
// after the sycl::program class removal to use the kernel_bundle instead.

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include "detail/context_impl.hpp"
#include "detail/kernel_program_cache.hpp"
#include "sycl/detail/pi.h"
#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <iostream>

using namespace sycl;

class CacheTestKernel {
public:
  void operator()(sycl::item<1>){};
};

class CacheTestKernel2 {
public:
  void operator()(sycl::item<1>){};
};

namespace sycl {
const static specialization_id<int> SpecConst1{42};
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<CacheTestKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "CacheTestKernel"; }
};
template <>
struct KernelInfo<CacheTestKernel2> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "CacheTestKernel2"; }
};
template <> const char *get_spec_constant_symbolic_ID<SpecConst1>() {
  return "SC1";
}
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  std::vector<char> SpecConstData;
  PiProperty SC1 = makeSpecConstant<int>(SpecConstData, "SC1", {0}, {0}, {42});

  PiPropertySet PropSet;
  addSpecConstants({SC1}, std::move(SpecConstData), PropSet);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({"CacheTestKernel", "CacheTestKernel2"});

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

class KernelAndProgramCacheTest : public ::testing::Test {
public:
  KernelAndProgramCacheTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineBefore<detail::PiApiKind::piKernelGetInfo>(
        redefinedKernelGetInfo);
  }

protected:
  unittest::PiMock Mock;
  sycl::platform Plt;
};

// Check that programs built from source are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_ProgramSourceNegativeBuild) {
  context Ctx{Plt};
  //   program Prg{Ctx};

  //   Prg.build_with_source("");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs built from source with options are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_ProgramSourceNegativeBuildWithOpts) {
  context Ctx{Plt};
  //   program Prg{Ctx};

  //   Prg.build_with_source("", "-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs compiled and linked from source are not cached.
TEST_F(KernelAndProgramCacheTest,
       DISABLED_ProgramSourceNegativeCompileAndLink) {
  context Ctx{Plt};
  //   program Prg{Ctx};

  //   Prg.compile_with_source("");
  //   Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that programs compiled and linked from source with options are not
// cached.
TEST_F(KernelAndProgramCacheTest,
       DISABLED_ProgramSourceNegativeCompileAndLinkWithOpts) {
  context Ctx{Plt};
  //   program Prg{Ctx};

  //   Prg.compile_with_source("");
  //   Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for source programs";
}

// Check that kernel_bundles with input_state are not cached.
TEST_F(KernelAndProgramCacheTest, KernelBundleInputState) {
  std::vector<sycl::device> Devices = Plt.get_devices();
  sycl::context Ctx(Devices[0]);

  auto KernelID1 = sycl::get_kernel_id<CacheTestKernel>();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {KernelID1});

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();

  EXPECT_EQ(Cache.size(), 0U)
    << "Expect empty cache for kernel_bundles build with input_state.";
}

// Check that kernel_bundles with object_state are not cached.
TEST_F(KernelAndProgramCacheTest, KernelBundleObjectState) {
  std::vector<sycl::device> Devices = Plt.get_devices();
  sycl::context Ctx(Devices[0]);

  auto KernelID1 = sycl::get_kernel_id<CacheTestKernel>();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::object>(Ctx, {KernelID1});

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();

  EXPECT_EQ(Cache.size(), 0U)
    << "Expect empty cache for kernel_bundles build with object_state.";
}

// Check that kernel_bundles with executable_state are cached.
TEST_F(KernelAndProgramCacheTest, KernelBundleExecutableState) {
  std::vector<sycl::device> Devices = Plt.get_devices();
  sycl::context Ctx(Devices[0]);

  auto KernelID1 = sycl::get_kernel_id<CacheTestKernel>();
  auto KernelID2 = sycl::get_kernel_id<CacheTestKernel2>();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {KernelID1});
  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {KernelID2});

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();

  EXPECT_EQ(Cache.size(), 1U)
    << "Expect non-empty cache for kernel_bundles with executable_state.";
}

// Check that kernel_bundle built with specialization constants are cached.
TEST_F(KernelAndProgramCacheTest, SpecConstantCacheNegative) {
  std::vector<sycl::device> Devices = Plt.get_devices();
  sycl::context Ctx(Devices[0]);

  auto KernelID1 = sycl::get_kernel_id<CacheTestKernel>();
  auto KernelID2 = sycl::get_kernel_id<CacheTestKernel2>();

  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {KernelID1});
  KernelBundle1.set_specialization_constant<SpecConst1>(80);
  sycl::build(KernelBundle1);
  EXPECT_EQ(KernelBundle1.get_specialization_constant<SpecConst1>(), 80)
      << "Wrong specialization constant";

  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {KernelID2});
  KernelBundle2.set_specialization_constant<SpecConst1>(70);
  sycl::build(KernelBundle2);
  EXPECT_EQ(KernelBundle2.get_specialization_constant<SpecConst1>(), 70)
      << "Wrong specialization constant";

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();

  EXPECT_EQ(Cache.size(), 2U) << "Expect an entry for each build in the cache.";
}

// Check that kernel_bundle created through join() is not cached.
TEST_F(KernelAndProgramCacheTest, KernelBundleJoin) {
  std::vector<sycl::device> Devices = Plt.get_devices();
  sycl::context Ctx(Devices[0]);

  auto KernelID1 = sycl::get_kernel_id<CacheTestKernel>();
  auto KernelID2 = sycl::get_kernel_id<CacheTestKernel2>();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {KernelID1});
  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {KernelID2});

  std::vector<kernel_bundle<sycl::bundle_state::executable>>
      KernelBundles {KernelBundle1, KernelBundle2};
  sycl::kernel_bundle KernelBundle3 = sycl::join(KernelBundles);

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();

  EXPECT_EQ(Cache.size(), 1U)
      << "Expect no caching for kennel_bundle created via join.";
}

// Check that programs built with options are cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_ProgramBuildPositiveBuildOpts) {
  context Ctx{Plt};
  //   program Prg1{Ctx};
  //   program Prg2{Ctx};
  //   program Prg3{Ctx};
  //   program Prg4{Ctx};
  //   program Prg5{Ctx};

  /* Build 5 instances of the same program. It is expected that there will be 3
   * instances of the program in the cache because Build of Prg1 is equal to
   * build of Prg5 and build of Prg2 is equal to build of Prg3.
   * */
  //   Prg1.build_with_kernel_type<CacheTestKernel>("-a");
  //   Prg2.build_with_kernel_type<CacheTestKernel>("-b");
  //   Prg3.build_with_kernel_type<CacheTestKernel>("-b");
  //   Prg4.build_with_kernel_type<CacheTestKernel>();
  //   Prg5.build_with_kernel_type<CacheTestKernel2>("-a");

  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 3U) << "Expect non-empty cache for programs";
}

// Check that programs built with compile options are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_ProgramBuildNegativeCompileOpts) {
  context Ctx{Plt};
  //   program Prg{Ctx};

  //   Prg.compile_with_kernel_type<CacheTestKernel>("-g");
  //   Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for programs";
}

// Check that programs built with link options are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_ProgramBuildNegativeLinkOpts) {
  context Ctx{Plt};
  //   program Prg{Ctx};

  //   Prg.compile_with_kernel_type<CacheTestKernel>();
  //   Prg.link("-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCache &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for programs";
}

// Check that kernels built without options are cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_KernelPositive) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.build_with_kernel_type<CacheTestKernel>();
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with options are cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_KernelPositiveBuildOpts) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.build_with_kernel_type<CacheTestKernel>("-g");

  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with compile options are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_KernelNegativeCompileOpts) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.compile_with_kernel_type<CacheTestKernel>("-g");
  //   Prg.link();
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels built with link options are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_KernelNegativeLinkOpts) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.compile_with_kernel_type<CacheTestKernel>();
  //   Prg.link("-g");
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels created from source are not cached.
TEST_F(KernelAndProgramCacheTest, DISABLED_KernelNegativeSource) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.build_with_source("");
  //   kernel Ker = Prg.get_kernel("test");

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
TEST_F(KernelAndProgramFastCacheTest, DISABLED_KernelPositive) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.build_with_kernel_type<CacheTestKernel>();
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with options are cached.
TEST_F(KernelAndProgramFastCacheTest, DISABLED_KernelPositiveBuildOpts) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.build_with_kernel_type<CacheTestKernel>("-g");

  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 1U) << "Expect non-empty cache for kernels";
}

// Check that kernels built with compile options are not cached.
TEST_F(KernelAndProgramFastCacheTest, DISABLED_KernelNegativeCompileOpts) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.compile_with_kernel_type<CacheTestKernel>("-g");
  //   Prg.link();
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels built with link options are not cached.
TEST_F(KernelAndProgramFastCacheTest, DISABLED_KernelNegativeLinkOpts) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.compile_with_kernel_type<CacheTestKernel>();
  //   Prg.link("-g");
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();
  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels are not cached if program is created from multiple
// programs.
TEST_F(KernelAndProgramFastCacheTest, DISABLED_KernelNegativeLinkedProgs) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg1{Ctx};
  //   program Prg2{Ctx};

  //   Prg1.compile_with_kernel_type<CacheTestKernel>();
  //   Prg2.compile_with_kernel_type<CacheTestKernel2>();
  //   program Prg({Prg1, Prg2});
  //   kernel Ker = Prg.get_kernel<CacheTestKernel>();

  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

// Check that kernels created from source are not cached.
TEST_F(KernelAndProgramFastCacheTest, DISABLED_KernelNegativeSource) {
  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  //   program Prg{Ctx};

  //   Prg.build_with_source("");
  //   kernel Ker = Prg.get_kernel("test");

  detail::KernelProgramCache::KernelFastCacheT &Cache =
      MockKernelProgramCache::getFastCache(CtxImpl->getKernelProgramCache());
  EXPECT_EQ(Cache.size(), 0U) << "Expect empty cache for kernels";
}

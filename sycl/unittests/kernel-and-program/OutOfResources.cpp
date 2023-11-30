//==--- OutOfResources.cpp --- OutOfResources unit test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include "detail/context_impl.hpp"
#include "detail/kernel_bundle_impl.hpp"
#include "detail/kernel_program_cache.hpp"
#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

class OutOfResourcesKernel1;
class OutOfResourcesKernel2;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<OutOfResourcesKernel1> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "OutOfResourcesKernel1"; }
};

template <>
struct KernelInfo<OutOfResourcesKernel2> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "OutOfResourcesKernel2"; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::PiImage makeImage(const char *kname) {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({kname});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Img[2] = {makeImage("OutOfResourcesKernel1"),
                                         makeImage("OutOfResourcesKernel2")};

static sycl::unittest::PiImageArray<2> ImgArray{Img};

static int nProgramCreate = 0;
static volatile bool outOfResourcesToggle = false;

static pi_result redefinedProgramCreate(pi_context context, const void *il,
                                        size_t length,
                                        pi_program *res_program) {
  ++nProgramCreate;
  if (outOfResourcesToggle) {
    outOfResourcesToggle = false;
    return PI_ERROR_OUT_OF_RESOURCES;
  }
  return PI_SUCCESS;
}

TEST(OutOfResourcesTest, foo) {
  sycl::unittest::PiMock Mock;
  Mock.redefineBefore<detail::PiApiKind::piProgramCreate>(
      redefinedProgramCreate);

  sycl::platform Plt{Mock.getPlatform()};
  sycl::context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  queue q(Ctx, default_selector_v);

  int runningTotal = 0;
  // Cache is empty, so one piProgramCreate call.
  q.single_task<class OutOfResourcesKernel1>([] {});
  EXPECT_EQ(nProgramCreate, runningTotal += 1);

  // Now, we make the next piProgramCreate call fail with
  // PI_ERROR_OUT_OF_RESOURCES. The caching mechanism should catch this,
  // clear the cache, and retry the piProgramCreate.
  outOfResourcesToggle = true;
  q.single_task<class OutOfResourcesKernel2>([] {});
  EXPECT_FALSE(outOfResourcesToggle);
  EXPECT_EQ(nProgramCreate, runningTotal += 2);
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 1U) << "Expect 1 program in cache";
  }

  // The next piProgramCreate call will fail with
  // PI_ERROR_OUT_OF_RESOURCES. But OutOfResourcesKernel2 is in
  // the cache, so we expect no new piProgramCreate calls.
  outOfResourcesToggle = true;
  q.single_task<class OutOfResourcesKernel2>([] {});
  EXPECT_TRUE(outOfResourcesToggle);
  EXPECT_EQ(nProgramCreate, runningTotal);

  // OutOfResourcesKernel1 is not in the cache, so we have to
  // build it. From what we set before, this call will fail,
  // the cache will clear out, and will try again.
  q.single_task<class OutOfResourcesKernel1>([] {});
  EXPECT_FALSE(outOfResourcesToggle);
  EXPECT_EQ(nProgramCreate, runningTotal += 2);
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 1U) << "Expect 1 program in cache";
  }

  // Finally, OutOfResourcesKernel1 will be in the cache, but
  // OutOfResourceKenel2 will not, so one more piProgramCreate.
  // Toggle is not set, so this should succeed.
  q.single_task<class OutOfResourcesKernel1>([] {});
  q.single_task<class OutOfResourcesKernel2>([] {});
  EXPECT_EQ(nProgramCreate, runningTotal += 1);
  {
    detail::KernelProgramCache::ProgramCache &Cache =
        CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
    EXPECT_EQ(Cache.size(), 2U) << "Expect 2 program in cache";
  }
}

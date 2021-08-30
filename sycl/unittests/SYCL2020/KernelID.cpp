//==------------ KernelID.cpp --- Kernel identifier unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class TestKernel1;
class TestKernel2;
class TestKernel3;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<TestKernel1> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "KernelID_TestKernel1"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

template <> struct KernelInfo<TestKernel2> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "KernelID_TestKernel2"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

template <> struct KernelInfo<TestKernel3> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "KernelID_TestKernel3"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

static sycl::unittest::PiImage
generateDefaultImage(std::initializer_list<std::string> Kernels) {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels(Kernels);

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage Imgs[2] = {
    generateDefaultImage({"KernelID_TestKernel1", "KernelID_TestKernel3"}),
    generateDefaultImage({"KernelID_TestKernel2"})};
static sycl::unittest::PiImageArray<2> ImgArray{Imgs};

TEST(KernelID, AllProgramKernelIds) {
  std::vector<sycl::kernel_id> AllKernelIDs = sycl::get_kernel_ids();

  // sycl::get_kernel_ids may pick up kernels from other unit tests, so there
  // must be *at least* the 3 from this test.
  EXPECT_GE(AllKernelIDs.size(), 3u);

  sycl::kernel_id TestKernel1ID = sycl::get_kernel_id<TestKernel1>();
  sycl::kernel_id TestKernel2ID = sycl::get_kernel_id<TestKernel2>();
  sycl::kernel_id TestKernel3ID = sycl::get_kernel_id<TestKernel3>();

  for (const sycl::kernel_id &TestKernelID :
       {TestKernel1ID, TestKernel2ID, TestKernel3ID}) {
    auto FoundKernelID =
        std::find(AllKernelIDs.begin(), AllKernelIDs.end(), TestKernelID);
    EXPECT_NE(FoundKernelID, AllKernelIDs.end());
  }
}

TEST(KernelID, FreeKernelIDEqualsKernelBundleId) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cout << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cout << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<sycl::kernel_id> BundleKernelIDs = KernelBundle.get_kernel_ids();

  sycl::kernel_id TestKernel1ID = sycl::get_kernel_id<TestKernel1>();
  sycl::kernel_id TestKernel2ID = sycl::get_kernel_id<TestKernel2>();
  sycl::kernel_id TestKernel3ID = sycl::get_kernel_id<TestKernel3>();

  for (const sycl::kernel_id &TestKernelID :
       {TestKernel1ID, TestKernel2ID, TestKernel3ID}) {
    auto FoundKernelID =
        std::find(BundleKernelIDs.begin(), BundleKernelIDs.end(), TestKernelID);
    EXPECT_NE(FoundKernelID, BundleKernelIDs.end());
  }
}

TEST(KernelID, KernelBundleKernelIDsIntersectAll) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cout << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cout << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<sycl::kernel_id> BundleKernelIDs = KernelBundle.get_kernel_ids();
  std::vector<sycl::kernel_id> AllKernelIDs = sycl::get_kernel_ids();

  for (const sycl::kernel_id &BundleKernelID : BundleKernelIDs) {
    auto FoundKernelID =
        std::find(AllKernelIDs.begin(), AllKernelIDs.end(), BundleKernelID);
    EXPECT_NE(FoundKernelID, AllKernelIDs.end());
  }
}

TEST(KernelID, KernelIDHasKernel) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cout << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::cuda) {
    std::cout << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<sycl::kernel_id> BundleKernelIDs = KernelBundle.get_kernel_ids();

  for (const sycl::kernel_id &BundleKernelID : BundleKernelIDs) {
    EXPECT_TRUE(KernelBundle.has_kernel(BundleKernelID));
  }

  sycl::kernel_id TestKernel1ID = sycl::get_kernel_id<TestKernel1>();
  sycl::kernel_id TestKernel2ID = sycl::get_kernel_id<TestKernel2>();
  sycl::kernel_id TestKernel3ID = sycl::get_kernel_id<TestKernel3>();

  std::vector<sycl::kernel_id> KernelIDs1 = {TestKernel1ID};
  auto InputBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs1);

  EXPECT_TRUE(InputBundle1.has_kernel(TestKernel1ID));
  EXPECT_FALSE(InputBundle1.has_kernel(TestKernel2ID));
  EXPECT_TRUE(InputBundle1.has_kernel(TestKernel3ID));

  std::vector<sycl::kernel_id> KernelIDs2 = {TestKernel2ID};
  auto InputBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs2);

  EXPECT_FALSE(InputBundle2.has_kernel(TestKernel1ID));
  EXPECT_TRUE(InputBundle2.has_kernel(TestKernel2ID));
  EXPECT_FALSE(InputBundle2.has_kernel(TestKernel3ID));

  std::vector<sycl::kernel_id> KernelIDs3 = {TestKernel3ID};
  auto InputBundle3 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs3);

  EXPECT_TRUE(InputBundle3.has_kernel(TestKernel1ID));
  EXPECT_FALSE(InputBundle3.has_kernel(TestKernel2ID));
  EXPECT_TRUE(InputBundle3.has_kernel(TestKernel3ID));

  std::vector<sycl::kernel_id> KernelIDs4 = {TestKernel1ID, TestKernel3ID};
  auto InputBundle4 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs4);

  EXPECT_TRUE(InputBundle4.has_kernel(TestKernel1ID));
  EXPECT_FALSE(InputBundle4.has_kernel(TestKernel2ID));
  EXPECT_TRUE(InputBundle4.has_kernel(TestKernel3ID));

  std::vector<sycl::kernel_id> KernelIDs5 = {TestKernel1ID, TestKernel2ID};
  auto InputBundle5 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs5);

  EXPECT_TRUE(InputBundle5.has_kernel(TestKernel1ID));
  EXPECT_TRUE(InputBundle5.has_kernel(TestKernel2ID));
  EXPECT_TRUE(InputBundle5.has_kernel(TestKernel3ID));

  std::vector<sycl::kernel_id> KernelIDs6 = {TestKernel2ID, TestKernel3ID};
  auto InputBundle6 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs6);

  EXPECT_TRUE(InputBundle6.has_kernel(TestKernel1ID));
  EXPECT_TRUE(InputBundle6.has_kernel(TestKernel2ID));
  EXPECT_TRUE(InputBundle6.has_kernel(TestKernel3ID));

  std::vector<sycl::kernel_id> KernelIDs7 = {TestKernel1ID, TestKernel2ID,
                                             TestKernel3ID};
  auto InputBundle7 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs7);

  EXPECT_TRUE(InputBundle7.has_kernel(TestKernel1ID));
  EXPECT_TRUE(InputBundle7.has_kernel(TestKernel2ID));
  EXPECT_TRUE(InputBundle7.has_kernel(TestKernel3ID));
}

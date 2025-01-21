//==------------ KernelID.cpp --- Kernel identifier unit tests -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class TestKernel1;
class TestKernel2;
class TestKernel3;
class ServiceKernel1;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernel1> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "KernelID_TestKernel1"; }
};

template <>
struct KernelInfo<TestKernel2> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "KernelID_TestKernel2"; }
};

template <>
struct KernelInfo<TestKernel3> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "KernelID_TestKernel3"; }
};

template <>
struct KernelInfo<ServiceKernel1> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "_ZTSN2cl4sycl6detail23__sycl_service_kernel__14ServiceKernel1";
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Imgs[2] = {
    sycl::unittest::generateDefaultImage(
        {"KernelID_TestKernel1", "KernelID_TestKernel3"}),
    sycl::unittest::generateDefaultImage(
        {"KernelID_TestKernel2",
         "_ZTSN2cl4sycl6detail23__sycl_service_kernel__14ServiceKernel1"})};
static sycl::unittest::MockDeviceImageArray<2> ImgArray{Imgs};

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

TEST(KernelID, NoServiceKernelIds) {
  const char *ServiceKernel1Name =
      sycl::detail::KernelInfo<ServiceKernel1>::getName();

  std::vector<sycl::kernel_id> AllKernelIDs = sycl::get_kernel_ids();

  auto NoFoundServiceKernelID = std::none_of(
      AllKernelIDs.begin(), AllKernelIDs.end(), [=](sycl::kernel_id KernelID) {
        return strcmp(KernelID.get_name(), ServiceKernel1Name) == 0;
      });

  EXPECT_TRUE(NoFoundServiceKernelID);
}

TEST(KernelID, FreeKernelIDEqualsKernelBundleId) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

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
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

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
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

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

TEST(KernelID, HasKernelTemplated) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_id TestKernel1ID = sycl::get_kernel_id<TestKernel1>();

  std::vector<sycl::kernel_id> KernelIDs1 = {TestKernel1ID};
  auto InputBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, KernelIDs1);

  EXPECT_TRUE(InputBundle1.has_kernel<TestKernel1>());
  EXPECT_FALSE(InputBundle1.has_kernel<TestKernel2>());
  EXPECT_TRUE(InputBundle1.has_kernel<TestKernel3>());
}

TEST(KernelID, GetKernelIDInvalidKernelName) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  try {
    sycl::get_kernel_id<class NotAKernel>();
    FAIL() << "Expected an exception";
  } catch (sycl::exception const &e) {
    EXPECT_TRUE(e.code() == sycl::errc::runtime);
    EXPECT_EQ(std::string("No kernel found with the specified name"), e.what());
  } catch (...) {
    FAIL() << "Expected sycl::exception";
  }
}

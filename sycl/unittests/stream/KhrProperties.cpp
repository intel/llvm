//==---- KhrProperties.cpp --- sycl_khr_properties stream constructor -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <type_traits>

using namespace sycl::khr;

class KhrStreamKernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<KhrStreamKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "KhrStreamKernel"; }
  static constexpr int64_t getKernelSize() { return sizeof(sycl::stream); }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"KhrStreamKernel"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

static_assert(is_property_list_for_v<empty_properties_t, sycl::stream>);
static_assert(std::is_constructible_v<sycl::stream, size_t, size_t,
                                      sycl::handler &, empty_properties_t>);

TEST(KhrStreamProperties, Constructor) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    sycl::stream S1(1024, 256, CGH, properties{});
    sycl::stream S2(2048, 128, CGH, empty_properties_t{});
    // Old-style construction must still resolve unambiguously.
    sycl::stream S3(512, 64, CGH);
    sycl::stream S4(512, 64, CGH, sycl::property_list{});
    EXPECT_EQ(S1.size(), 1024u);
    EXPECT_EQ(S1.get_work_item_buffer_size(), 256u);
    EXPECT_EQ(S2.size(), 2048u);
    EXPECT_EQ(S2.get_work_item_buffer_size(), 128u);
    CGH.single_task<KhrStreamKernel>([=]() { S1 << "Hello" << sycl::endl; });
  });
  Q.wait();
}

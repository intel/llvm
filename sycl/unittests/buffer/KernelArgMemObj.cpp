//==----------- KernelArgMemObj.cpp ---- Scheduler unit tests ---------- ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

class TestKernelWithMemObj;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernelWithMemObj> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernelWithMemObj"; }
  static constexpr unsigned getNumParams() { return 1; }
  static constexpr const detail::kernel_param_desc_t &getParamDesc(int) {
    return desc;
  }
  static constexpr uint32_t getKernelSize() { return 32; }

private:
  static constexpr detail::kernel_param_desc_t desc{
      detail::kernel_param_kind_t::kind_accessor,
      int(access::target::device) /*info*/, 0 /*offset*/};
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"TestKernelWithMemObj"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

using namespace sycl;

bool PropertyPresent = false;
ur_mem_flags_t MemFlags{};

ur_result_t redefinedEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  auto Args = *params.ppArgs;
  for (uint32_t i = 0; i < *params.pnumArgs; i++) {
    if (Args[i].type != UR_EXP_KERNEL_ARG_TYPE_MEM_OBJ) {
      continue;
    }
    PropertyPresent = Args[i].value.memObjTuple.flags != 0;
    if (PropertyPresent) {
      MemFlags = Args[i].value.memObjTuple.flags;
    }
  }
  return UR_RESULT_SUCCESS;
}

class BuferTestUrArgs : public ::testing::Test {
public:
  BuferTestUrArgs() : Mock(), Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    PropertyPresent = false;
    MemFlags = 0;
    mock::getCallbacks().set_before_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &redefinedEnqueueKernelLaunchWithArgsExp);
  }

  template <sycl::access::mode AccessMode>
  void TestFunc(ur_mem_flags_t ExpectedAccessMode) {
    queue Queue(context(Plt), default_selector_v);
    sycl::buffer<int, 1> Buf(3);
    Queue
        .submit([&](sycl::handler &cgh) {
          auto acc = Buf.get_access<AccessMode>(cgh);
          cgh.single_task<TestKernelWithMemObj>([=]() {
            if constexpr (AccessMode != sycl::access::mode::read)
              acc[0] = 4;
            else
              std::ignore = acc[0];
          });
        })
        .wait();
    ASSERT_TRUE(PropertyPresent);
    EXPECT_EQ(MemFlags, ExpectedAccessMode);
  }

protected:
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  sycl::platform Plt;
};

TEST_F(BuferTestUrArgs, KernelSetArgMemObjReadWrite) {
  TestFunc<sycl::access::mode::read_write>(UR_MEM_FLAG_READ_WRITE);
}

TEST_F(BuferTestUrArgs, KernelSetArgMemObjDiscardReadWrite) {
  TestFunc<sycl::access::mode::discard_read_write>(UR_MEM_FLAG_READ_WRITE);
}

TEST_F(BuferTestUrArgs, KernelSetArgMemObjRead) {
  TestFunc<sycl::access::mode::read>(UR_MEM_FLAG_READ_ONLY);
}

TEST_F(BuferTestUrArgs, KernelSetArgMemObjWrite) {
  TestFunc<sycl::access::mode::write>(UR_MEM_FLAG_WRITE_ONLY);
}

TEST_F(BuferTestUrArgs, KernelSetArgMemObjDiscardWrite) {
  TestFunc<sycl::access::mode::discard_write>(UR_MEM_FLAG_WRITE_ONLY);
}

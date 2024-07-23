//==----------- KernelArgMemObj.cpp ---- Scheduler unit tests ---------- ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

class TestKernelWithMemObj;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestKernelWithMemObj> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestKernelWithMemObj"; }
  static constexpr unsigned getNumParams() { return 1; }
  static const detail::kernel_param_desc_t &getParamDesc(int) {
    static detail::kernel_param_desc_t desc{
        detail::kernel_param_kind_t::kind_accessor,
        int(access::target::device) /*info*/, 0 /*offset*/};
    return desc;
  }
  static constexpr uint32_t getKernelSize() { return 32; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static auto Img =
    sycl::unittest::generateDefaultImage({"TestKernelWithMemObj"});
static sycl::unittest::PiImageArray<1> ImgArray{&Img};

using namespace sycl;

bool PropertyPresent = false;
pi_mem_obj_property PropsCopy{};

pi_result redefinedKernelSetArgMemObj(pi_kernel kernel, pi_uint32 arg_index,
                                      const pi_mem_obj_property *arg_properties,
                                      const pi_mem *arg_value) {
  PropertyPresent = arg_properties != nullptr;
  if (PropertyPresent)
    PropsCopy = *arg_properties;
  return PI_SUCCESS;
}

class BufferTestPiArgs : public ::testing::Test {
public:
  BufferTestPiArgs()
      : Mock(sycl::backend::ext_oneapi_level_zero), Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    PropertyPresent = false;
    PropsCopy = {};
    Mock.redefineBefore<detail::PiApiKind::piextKernelSetArgMemObj>(
        redefinedKernelSetArgMemObj);
  }

  template <sycl::access::mode AccessMode>
  void TestFunc(pi_mem_obj_access ExpectedAccessMode) {
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
    EXPECT_EQ(PropsCopy.type, PI_KERNEL_ARG_MEM_OBJ_ACCESS);
    EXPECT_EQ(PropsCopy.mem_access, ExpectedAccessMode);
  }

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
};

TEST_F(BufferTestPiArgs, KernelSetArgMemObjReadWrite) {
  TestFunc<sycl::access::mode::read_write>(PI_ACCESS_READ_WRITE);
}

TEST_F(BufferTestPiArgs, KernelSetArgMemObjDiscardReadWrite) {
  TestFunc<sycl::access::mode::discard_read_write>(PI_ACCESS_READ_WRITE);
}

TEST_F(BufferTestPiArgs, KernelSetArgMemObjRead) {
  TestFunc<sycl::access::mode::read>(PI_ACCESS_READ_ONLY);
}

TEST_F(BufferTestPiArgs, KernelSetArgMemObjWrite) {
  TestFunc<sycl::access::mode::write>(PI_ACCESS_WRITE_ONLY);
}

TEST_F(BufferTestPiArgs, KernelSetArgMemObjDiscardWrite) {
  TestFunc<sycl::access::mode::discard_write>(PI_ACCESS_WRITE_ONLY);
}

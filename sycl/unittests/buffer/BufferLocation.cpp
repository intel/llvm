//==-------- buffer_location.cpp --- check buffer_location property --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <detail/buffer_impl.hpp>

const uint64_t DEFAULT_VALUE = 7777;
static uint64_t PassedLocation = DEFAULT_VALUE;

ur_result_t redefinedMemBufferCreateBefore(void *pParams) {
  auto params = reinterpret_cast<ur_mem_buffer_create_params_t *>(pParams);
  PassedLocation = DEFAULT_VALUE;
  if (!*params->ppProperties)
    return UR_RESULT_SUCCESS;

  auto nextProps =
      static_cast<ur_base_properties_t *>((*params->ppProperties)->pNext);

  // properties must ended by 0
  while (nextProps) {
    if (nextProps->stype !=
        UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES) {
      nextProps = static_cast<ur_base_properties_t *>(nextProps->pNext);
      break;
    }
    PassedLocation =
        reinterpret_cast<ur_buffer_alloc_location_properties_t *>(nextProps)
            ->location;
    nextProps = reinterpret_cast<ur_base_properties_t *>(nextProps->pNext);
  }

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_DEVICE_INFO_TYPE: {
    auto *Result = reinterpret_cast<ur_device_type_t *>(*params->ppPropValue);
    *Result = UR_DEVICE_TYPE_FPGA;
    break;
  }
  case UR_DEVICE_INFO_COMPILER_AVAILABLE: {
    auto *Result = reinterpret_cast<ur_bool_t *>(*params->ppPropValue);
    *Result = true;
    break;
  }
  case UR_DEVICE_INFO_EXTENSIONS: {
    const std::string name = "cl_intel_mem_alloc_buffer_location";

    // Increase size by one for the null terminator
    const size_t nameSize = name.size() + 1;

    if (!*params->ppPropValue) {
      // Choose bigger size so that both original and redefined function
      // has enough memory for storing the extension string
      **params->ppPropSizeRet = nameSize > **params->ppPropSizeRet
                                    ? nameSize
                                    : **params->ppPropSizeRet;
    } else {
      char *dst = static_cast<char *>(*params->ppPropValue);
      strcpy(dst, name.data());
    }
    break;
  }
  // This mock device has no sub-devices
  case UR_DEVICE_INFO_SUPPORTED_PARTITIONS: {
    if (*params->ppPropSizeRet) {
      **params->ppPropSizeRet = 0;
    }
    break;
  }
  case UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    assert(*params->ppropSize == sizeof(ur_device_affinity_domain_flags_t));
    if (*params->ppPropValue) {
      *static_cast<ur_device_affinity_domain_flags_t *>(*params->ppPropValue) =
          0;
    }
    break;
  }
  default:
    break;
  }
  return UR_RESULT_SUCCESS;
}

class BufferTest : public ::testing::Test {
public:
  BufferTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                             &redefinedMemBufferCreateBefore);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &redefinedDeviceGetInfoAfter);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt;
};

// Test that buffer_location was passed correctly
TEST_F(BufferTest, BufferLocationOnly) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list<
            sycl::ext::intel::property::buffer_location::instance<2>>
            PL{sycl::ext::intel::buffer_location<2>};
        sycl::accessor<
            int, 1, sycl::access::mode::read_write,
            sycl::access::target::global_buffer,
            sycl::access::placeholder::false_t,
            sycl::ext::oneapi::accessor_property_list<
                sycl::ext::intel::property::buffer_location::instance<2>>>
            Acc{Buf, cgh, sycl::read_write, PL};
        constexpr size_t KS = sizeof(decltype(Acc));
        cgh.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, (uint64_t)2);
}

// Test that buffer_location was passed correcty if there is one more accessor
// property and buffer_location is correctly chaned by creating new accessors
TEST_F(BufferTest, BufferLocationWithAnotherProp) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list<
            sycl::ext::oneapi::property::no_alias::instance<true>,
            sycl::ext::intel::property::buffer_location::instance<5>>
            PL{sycl::ext::oneapi::no_alias,
               sycl::ext::intel::buffer_location<5>};
        sycl::accessor<
            int, 1, sycl::access::mode::write,
            sycl::access::target::global_buffer,
            sycl::access::placeholder::false_t,
            sycl::ext::oneapi::accessor_property_list<
                sycl::ext::oneapi::property::no_alias::instance<true>,
                sycl::ext::intel::property::buffer_location::instance<5>>>
            Acc{Buf, cgh, sycl::write_only, PL};

        constexpr size_t KS = sizeof(decltype(Acc));
        cgh.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, (uint64_t)5);

  // Check that if new accessor created, buffer_location is changed
  Queue
      .submit([&](sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list<
            sycl::ext::intel::property::buffer_location::instance<3>>
            PL{sycl::ext::intel::buffer_location<3>};
        sycl::accessor<
            int, 1, sycl::access::mode::write,
            sycl::access::target::global_buffer,
            sycl::access::placeholder::false_t,
            sycl::ext::oneapi::accessor_property_list<
                sycl::ext::intel::property::buffer_location::instance<3>>>
            Acc{Buf, cgh, sycl::write_only, PL};
      })
      .wait();
  std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
      sycl::detail::getSyclObjImpl(Buf);
  EXPECT_EQ(
      BufImpl->get_property<sycl::property::buffer::detail::buffer_location>()
          .get_buffer_location(),
      (uint64_t)3);

  // Check that if new accessor created, buffer_location is deleted from buffer
  Queue
      .submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::write,
                       sycl::access::target::global_buffer,
                       sycl::access::placeholder::false_t,
                       sycl::ext::oneapi::accessor_property_list<>>
            Acc{Buf, cgh, sycl::write_only};
      })
      .wait();

  EXPECT_EQ(
      BufImpl->has_property<sycl::property::buffer::detail::buffer_location>(),
      0);
}

// Test that there is no buffer_location property
TEST_F(BufferTest, WOBufferLocation) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access::mode::read_write,
                       sycl::access::target::global_buffer,
                       sycl::access::placeholder::false_t,
                       sycl::ext::oneapi::accessor_property_list<>>
            Acc{Buf, cgh, sycl::read_write};
        constexpr size_t KS = sizeof(decltype(Acc));
        cgh.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, DEFAULT_VALUE);
}

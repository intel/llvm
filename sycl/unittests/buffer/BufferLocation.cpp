//==-------- buffer_location.cpp --- check buffer_location property --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>

#include <gtest/gtest.h>

const uint64_t DEFAULT_VALUE = 7777;
static uint64_t PassedLocation = DEFAULT_VALUE;

pi_result redefinedMemBufferCreate(pi_context, pi_mem_flags, size_t size,
                                   void *, pi_mem *,
                                   const pi_mem_properties *properties) {
  PassedLocation = DEFAULT_VALUE;
  if (!properties)
    return PI_SUCCESS;

  // properties must ended by 0
  size_t I = 0;
  while (true) {
    if (properties[I] != 0) {
      if (properties[I] != PI_MEM_PROPERTIES_ALLOC_BUFFER_LOCATION) {
        I += 2;
      } else {
        PassedLocation = properties[I + 1];
        break;
      }
    }
  }

  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_ACC;
  }
  if (param_name == PI_DEVICE_INFO_COMPILER_AVAILABLE) {
    auto *Result = reinterpret_cast<pi_bool *>(param_value);
    *Result = true;
  }
  if (param_name == PI_DEVICE_INFO_EXTENSIONS) {
    const std::string name = "cl_intel_mem_alloc_buffer_location";
    if (!param_value) {
      *param_value_size_ret = name.size();
    } else {
      char *dst = static_cast<char *>(param_value);
      strcpy(dst, name.data());
    }
  }
  return PI_SUCCESS;
}

class BufferTest : public ::testing::Test {
public:
  BufferTest() : Plt{sycl::default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host() || Plt.get_backend() != sycl::backend::opencl) {
      std::cout << "This test is only supported on OpenCL backend\n";
      std::cout << "Current platform is "
                << Plt.get_info<sycl::info::platform::name>();
      return;
    }

    Mock = std::make_unique<sycl::unittest::PiMock>(Plt);

    setupDefaultMockAPIs(*Mock);
    Mock->redefine<sycl::detail::PiApiKind::piMemBufferCreate>(
        redefinedMemBufferCreate);
    Mock->redefine<sycl::detail::PiApiKind::piDeviceGetInfo>(
        redefinedDeviceGetInfo);
  }

protected:
  std::unique_ptr<sycl::unittest::PiMock> Mock;
  sycl::platform Plt;
};

// Test that buffer_location was passed correctly
TEST_F(BufferTest, BufferLocationOnly) {
  if (Plt.is_host() || Plt.get_backend() != sycl::backend::opencl) {
    return;
  }

  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  cl::sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list<
            cl::sycl::ext::intel::property::buffer_location::instance<2>>
            PL{sycl::ext::intel::buffer_location<2>};
        sycl::accessor<
            int, 1, cl::sycl::access::mode::read_write,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::false_t,
            cl::sycl::ext::oneapi::accessor_property_list<
                cl::sycl::ext::intel::property::buffer_location::instance<2>>>
            Acc{Buf, cgh, sycl::read_write, PL};
        cgh.single_task<TestKernel>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, (uint64_t)2);
}

// Test that buffer_location was passed correcty if there is one more accessor
// property and buffer_location is correctly chaned by creating new accessors
TEST_F(BufferTest, BufferLocationWithAnotherProp) {
  if (Plt.is_host() || Plt.get_backend() != sycl::backend::opencl) {
    return;
  }

  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  cl::sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list<
            cl::sycl::ext::oneapi::property::no_alias::instance<true>,
            cl::sycl::ext::intel::property::buffer_location::instance<5>>
            PL{sycl::ext::oneapi::no_alias,
               sycl::ext::intel::buffer_location<5>};
        sycl::accessor<
            int, 1, cl::sycl::access::mode::write,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::false_t,
            cl::sycl::ext::oneapi::accessor_property_list<
                cl::sycl::ext::oneapi::property::no_alias::instance<true>,
                cl::sycl::ext::intel::property::buffer_location::instance<5>>>
            Acc{Buf, cgh, sycl::write_only, PL};

        cgh.single_task<TestKernel>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, (uint64_t)5);

  // Check that if new accessor created, buffer_location is changed
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list<
            cl::sycl::ext::intel::property::buffer_location::instance<3>>
            PL{sycl::ext::intel::buffer_location<3>};
        sycl::accessor<
            int, 1, cl::sycl::access::mode::write,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::false_t,
            cl::sycl::ext::oneapi::accessor_property_list<
                cl::sycl::ext::intel::property::buffer_location::instance<3>>>
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
      .submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::false_t,
                       cl::sycl::ext::oneapi::accessor_property_list<>>
            Acc{Buf, cgh, sycl::write_only};
      })
      .wait();

  EXPECT_EQ(
      BufImpl->has_property<sycl::property::buffer::detail::buffer_location>(),
      0);
}

// Test that there is no buffer_location property
TEST_F(BufferTest, WOBufferLocation) {
  if (Plt.is_host() || Plt.get_backend() != sycl::backend::opencl) {
    return;
  }

  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  cl::sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::false_t,
                       cl::sycl::ext::oneapi::accessor_property_list<>>
            Acc{Buf, cgh, sycl::read_write};
        cgh.single_task<TestKernel>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, DEFAULT_VALUE);
}

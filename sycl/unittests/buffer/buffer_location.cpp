//==-------- buffer_location.cpp --- check buffer_location property --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

  // properties must be ended by 0
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

class BufferTest : public ::testing::Test {
public:
  BufferTest() : Plt{sycl::default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host() || Plt.get_backend() != sycl::backend::opencl) {
      return;
    }

    Mock = std::make_unique<sycl::unittest::PiMock>(Plt);

    setupDefaultMockAPIs(*Mock);
    Mock->redefine<sycl::detail::PiApiKind::piMemBufferCreate>(
        redefinedMemBufferCreate);
  }

protected:
  std::unique_ptr<sycl::unittest::PiMock> Mock;
  sycl::platform Plt;
};

// Test that buffer_location was passed correctly
TEST_F(BufferTest, BufferLocationOnly) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::default_selector{}};

  const uint64_t BUFFER_LOCATION = 2;
  cl::sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list PL{
            sycl::ext::intel::buffer_location<BUFFER_LOCATION>};
        sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::false_t,
                       cl::sycl::ext::oneapi::accessor_property_list<
                           cl::sycl::ext::intel::property::buffer_location::
                               instance<BUFFER_LOCATION>>>
            Acc{Buf, cgh, sycl::read_write, PL};
        cgh.single_task<TestKernel>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, BUFFER_LOCATION);
}

// Test that buffer_location was passed correcty if there is one more accessor
// property and buffer_location is correctly chaned by creating new accessors
TEST_F(BufferTest, BufferLocationWithAnotherProp) {
  const uint64_t BUFFER_LOCATION = 5;
  const uint64_t BUFFER_LOCATION2 = 3;

  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::default_selector{}};

  cl::sycl::buffer<int, 1> Buf(3);
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list PL{
            sycl::ext::oneapi::no_alias,
            sycl::ext::intel::buffer_location<BUFFER_LOCATION>};
        sycl::accessor<
            int, 1, cl::sycl::access::mode::write,
            cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder::false_t,
            cl::sycl::ext::oneapi::accessor_property_list<
                cl::sycl::ext::oneapi::property::no_alias::instance<true>,
                cl::sycl::ext::intel::property::buffer_location::instance<
                    BUFFER_LOCATION>>>
            Acc{Buf, cgh, sycl::write_only, PL};

        cgh.single_task<TestKernel>([=]() { Acc[0] = 4; });
      })
      .wait();
  EXPECT_EQ(PassedLocation, BUFFER_LOCATION);

  // Check that if new accessor created, buffer_location is changed
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list PL{
            sycl::ext::intel::buffer_location<BUFFER_LOCATION2>};
        sycl::accessor<int, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::false_t,
                       cl::sycl::ext::oneapi::accessor_property_list<
                           cl::sycl::ext::intel::property::buffer_location::
                               instance<BUFFER_LOCATION2>>>
            Acc{Buf, cgh, sycl::write_only, PL};
      })
      .wait();
  std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
      sycl::detail::getSyclObjImpl(Buf);
  EXPECT_EQ(
      BufImpl->get_property<sycl::property::buffer::detail::buffer_location>()
          .get_buffer_location(),
      BUFFER_LOCATION2);

  // Check that if new accessor created, buffer_location is deleted from buffer
  Queue
      .submit([&](cl::sycl::handler &cgh) {
        sycl::ext::oneapi::accessor_property_list PL{
            sycl::ext::oneapi::no_alias, sycl::ext::intel::buffer_location<1>};
        sycl::accessor<int, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::false_t,
                       cl::sycl::ext::oneapi::accessor_property_list<>>
            Acc{Buf, cgh, sycl::write_only};
      })
      .wait();
  // std::shared_ptr<sycl::detail::buffer_impl> BufImpl =
  // sycl::detail::getSyclObjImpl(Buf);
  EXPECT_EQ(
      BufImpl->has_property<sycl::property::buffer::detail::buffer_location>(),
      0);
}

// Test that there is no buffer_location property
TEST_F(BufferTest, WOBufferLocation) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::default_selector{}};

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

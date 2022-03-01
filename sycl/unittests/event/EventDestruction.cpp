//==------- EventDestruction.cpp --- Check correct event destruction -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace sycl;

static int ReleaseCounter = 0;
static pi_result redefinedEventRelease(pi_event event) {
  ++ReleaseCounter;
  return PI_SUCCESS;
}

pi_result redefinedMemBufferCreate(pi_context, pi_mem_flags, size_t size,
                                   void *, pi_mem *,
                                   const pi_mem_properties *) {
  return PI_SUCCESS;
}

class EventDestructionTest : public ::testing::Test {
public:
  EventDestructionTest() : Plt{default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      std::cout << "This test is only supported on OpenCL backend\n";
      std::cout << "Current platform is "
                << Plt.get_info<sycl::info::platform::name>();
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);

    setupDefaultMockAPIs(*Mock);
    Mock->redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);
    Mock->redefine<sycl::detail::PiApiKind::piMemBufferCreate>(
        redefinedMemBufferCreate);
  }

protected:
  std::unique_ptr<unittest::PiMock> Mock;
  platform Plt;
};

// Test that events are destructed in correct time
TEST_F(EventDestructionTest, EventDestruction) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::default_selector{}};

  {
    ReleaseCounter = 0;
    sycl::event E1{};

    {
      sycl::event E0 = Queue.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<TestKernel>([]() {});
      });
      E1 = Queue.submit([&](cl::sycl::handler &cgh) {
        cgh.depends_on(E0);
        cgh.single_task<TestKernel>([]() {});
      });
      E1.wait();
    }
    EXPECT_EQ(ReleaseCounter, 0);

    sycl::event E2 = Queue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on(E1);
      cgh.single_task<TestKernel>([]() {});
    });
    E2.wait();
    EXPECT_EQ(ReleaseCounter, 1);

    sycl::event E3 = Queue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on({E1, E2});
      cgh.single_task<TestKernel>([]() {});
    });
    E3.wait();
    EXPECT_EQ(ReleaseCounter, 1);
  }

  {
    ReleaseCounter = 0;
    int data[2] = {0, 1};
    sycl::buffer<int, 1> Buf(&data[0], sycl::range<1>(2));
    Queue.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel>([=]() {});
    });

    Queue.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel>([=]() {});
    });
    sycl::event E1 = Queue.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel>([=]() {});
    });
    sycl::event E2 = Queue.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel>([=]() {});
    });
    E2.wait();
    EXPECT_EQ(ReleaseCounter, 1);
  }
}

// Test for event::get_wait_list
TEST_F(EventDestructionTest, GetWaitList) {
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    return;
  }
  ReleaseCounter = 0;
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::default_selector{}};
  // Test for get_wait_list with host_task
  {
    sycl::event eA =
        Queue.submit([&](sycl::handler &cgh) { cgh.host_task([]() {}); });
    sycl::event eB = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(eA);
      cgh.host_task([]() {});
    });

    auto wait_list = eB.get_wait_list();
    assert(wait_list.size() == 1);
    ASSERT_EQ(wait_list[0], eA);

    sycl::event eC = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on({eA, eB});
      cgh.host_task([]() {});
    });

    wait_list = eC.get_wait_list();
    assert(wait_list.size() == 2);
    ASSERT_EQ(wait_list[0], eA);
    ASSERT_EQ(wait_list[1], eB);

    eC.wait();
  }

  // Test for get_wait_list with single_task
  {
    sycl::event E1{};

    {
      sycl::event E0 = Queue.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<TestKernel>([]() {});
      });
      E1 = Queue.submit([&](cl::sycl::handler &cgh) {
        cgh.depends_on(E0);
        cgh.single_task<TestKernel>([]() {});
      });
      E1.wait();
      auto wait_list = E1.get_wait_list();
      ASSERT_EQ(wait_list.size(), (size_t)1);
      ASSERT_EQ(wait_list[0], E0);
    }

    auto wait_list = E1.get_wait_list();
    ASSERT_EQ(wait_list.size(), (size_t)1);
    EXPECT_EQ(ReleaseCounter, 0);
    wait_list.clear();

    sycl::event E2 = Queue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on(E1);
      cgh.single_task<TestKernel>([]() {});
    });
    E2.wait();

    sycl::event E3 = Queue.submit([&](cl::sycl::handler &cgh) {
      cgh.depends_on({E1, E2});
      cgh.single_task<TestKernel>([]() {});
    });
    E3.wait();

    wait_list = E3.get_wait_list();
    ASSERT_EQ(wait_list.size(), (size_t)2);
    ASSERT_EQ(wait_list[0], E1);
    ASSERT_EQ(wait_list[1], E2);
  }
}

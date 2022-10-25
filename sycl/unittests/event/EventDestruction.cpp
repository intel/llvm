//==------- EventDestruction.cpp --- Check correct event destruction -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  EventDestructionTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineBefore<detail::PiApiKind::piEventRelease>(
        redefinedEventRelease);
    Mock.redefineBefore<detail::PiApiKind::piMemBufferCreate>(
        redefinedMemBufferCreate);
  }

protected:
  unittest::PiMock Mock;
  sycl::platform Plt;
};

// Test that events are destructed in correct time
TEST_F(EventDestructionTest, EventDestruction) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::default_selector{}};

  {
    ReleaseCounter = 0;
    sycl::event E1{};

    {
      sycl::event E0 = Queue.submit(
          [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
      E1 = Queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(E0);
        cgh.single_task<TestKernel<>>([]() {});
      });
      E1.wait();
    }
    // When a command is cleared we clear now only dependencies of the
    // dependencies of the associated event. So, when the command
    // associated with E0 event is destroyed, this event is still in
    // E1 dependencies, which will not be cleared.
    // Therefore no event release should be called until here.
    EXPECT_EQ(ReleaseCounter, 0);

    sycl::event E2 = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(E1);
      cgh.single_task<TestKernel<>>([]() {});
    });
    E2.wait();
    // Dependencies of E1 should be cleared here. It depends on E0.
    EXPECT_EQ(ReleaseCounter, 1);

    sycl::event E3 = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on({E1, E2});
      cgh.single_task<TestKernel<>>([]() {});
    });
    E3.wait();
    // Dependency of E1 has already cleared. E2 depends on E1 that
    // can't be cleared yet.
    EXPECT_EQ(ReleaseCounter, 1);
  }

  {
    ReleaseCounter = 0;
    int data[2] = {0, 1};
    sycl::buffer<int, 1> Buf(&data[0], sycl::range<1>(2));
    Queue.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel<>>([=]() {});
    });

    Queue.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel<>>([=]() {});
    });
    sycl::event E1 = Queue.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel<>>([=]() {});
    });
    sycl::event E2 = Queue.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<TestKernel<>>([=]() {});
    });
    E2.wait();
    // Dependencies are deleted through one level of dependencies. When
    // fourth command group is submitted the destructor of third command
    // is called. It depends on second command, so dependencies of second
    // command will be cleared. It leads to release event associated with
    // first command
    EXPECT_EQ(ReleaseCounter, 1);
  }
}

// Test for event::get_wait_list
TEST_F(EventDestructionTest, GetWaitList) {
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
    ASSERT_EQ(wait_list.size(), (size_t)1);
    ASSERT_EQ(wait_list[0], eA);

    sycl::event eC = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on({eA, eB});
      cgh.host_task([]() {});
    });

    wait_list = eC.get_wait_list();
    ASSERT_EQ(wait_list.size(), (size_t)2);
    ASSERT_EQ(wait_list[0], eA);
    ASSERT_EQ(wait_list[1], eB);

    eC.wait();
  }

  // Test for get_wait_list with single_task
  {
    sycl::event E1{};

    {
      sycl::event E0 = Queue.submit(
          [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
      E1 = Queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(E0);
        cgh.single_task<TestKernel<>>([]() {});
      });
      E1.wait();
      auto wait_list = E1.get_wait_list();
      ASSERT_EQ(wait_list.size(), (size_t)1);
      ASSERT_EQ(wait_list[0], E0);
    }

    auto wait_list = E1.get_wait_list();
    ASSERT_EQ(wait_list.size(), (size_t)1);
    EXPECT_EQ(ReleaseCounter, 0);

    sycl::event E2 = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(E1);
      cgh.single_task<TestKernel<>>([]() {});
    });
    E2.wait();

    sycl::event E3 = Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on({E1, E2});
      cgh.single_task<TestKernel<>>([]() {});
    });
    E3.wait();

    wait_list = E3.get_wait_list();
    ASSERT_EQ(wait_list.size(), (size_t)2);
    ASSERT_EQ(wait_list[0], E1);
    ASSERT_EQ(wait_list[1], E2);
  }
}

/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  Device.cpp
 *
 *  Description:
 *    Device info and selection tests
 **************************************************************************/

// The original source was under the license below:
//===-- Device.cpp -  -*- C++ -* ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <syclcompat/device.hpp>

/*
  Device Tests
*/

class DeviceTestsFixt : public ::testing::Test {
protected:
  unsigned int n_devices{};
  sycl::queue def_q_;

public:
  DeviceTestsFixt()
      : n_devices{syclcompat::detail::dev_mgr::instance().device_count()},
        def_q_{syclcompat::get_default_queue()} {}
};

TEST_F(DeviceTestsFixt, AtLeastOneDevice) { EXPECT_GT(n_devices, 0); }

// Check multiple threads get same device by default
TEST(Device, Threads) {
  unsigned int thread_dev_id{};
  std::thread other_thread{
      [&]() { thread_dev_id = syclcompat::get_current_device_id(); }};
  other_thread.join();
  EXPECT_EQ(thread_dev_id, syclcompat::get_current_device_id());
}

// Check a thread is able to select a non-default device
TEST_F(DeviceTestsFixt, DeviceSelect) {
  if (n_devices == 1)
    GTEST_SKIP(); // Can only test this w/ multiple devices

  constexpr unsigned int TARGET_DEV = 1;
  unsigned int thread_dev_id{};
  std::thread other_thread{[&]() {
    syclcompat::select_device(TARGET_DEV);
    thread_dev_id = syclcompat::get_current_device_id();
  }};
  other_thread.join();
  EXPECT_EQ(thread_dev_id, TARGET_DEV);
}

// Check the device returned matches the device ID
TEST(Device, MatchesID) {
  EXPECT_EQ(syclcompat::get_device(syclcompat::get_current_device_id()),
            syclcompat::get_current_device());
}

// Check error on insufficient devices
TEST_F(DeviceTestsFixt, NotEnoughDevices) {
  EXPECT_THROW(syclcompat::select_device(n_devices), std::runtime_error);
}

// Check the default context matches default queue's context
TEST_F(DeviceTestsFixt, DefaultContext) {
  EXPECT_EQ(def_q_.get_context(), syclcompat::get_default_context());
}

/*
  Queue Tests
*/

class DefaultQueueTest : public ::testing::Test {
protected:
  sycl::queue q_;

public:
  DefaultQueueTest() : q_{syclcompat::get_default_queue()} {}
};

TEST_F(DefaultQueueTest, MakeInOrderQueue) { EXPECT_TRUE(q_.is_in_order()); }

TEST_F(DefaultQueueTest, CheckDefaultDevice) {
  EXPECT_EQ(q_.get_device(), sycl::device{sycl::default_selector_v});
}

// Check behaviour of in order & out of order queue construction
TEST(Queues, QueuePropOrder) {
  sycl::queue q_create_def{syclcompat::create_queue()};
  EXPECT_TRUE(q_create_def.is_in_order());
  sycl::queue q_in_order{syclcompat::create_queue(false, true)};
  EXPECT_TRUE(q_in_order.is_in_order());
  sycl::queue q_out_order{syclcompat::create_queue(false, false)};
  EXPECT_FALSE(q_out_order.is_in_order());
}

class DeviceExtFixt : public ::testing::Test {
protected:
  syclcompat::device_ext &dev_;

public:
  DeviceExtFixt() : dev_{syclcompat::get_current_device()} {}

  void SetUp() { dev_.reset(); }
};

TEST_F(DeviceExtFixt, DeviceExtAPI) {
  dev_.is_native_host_atomic_supported();
  dev_.get_major_version();
  dev_.get_minor_version();
  dev_.get_max_compute_units();
  dev_.get_max_clock_frequency();
  dev_.get_integrated();
  syclcompat::device_info Info;
  dev_.get_device_info(Info);
  Info = dev_.get_device_info();
  dev_.reset();
  auto QueuePtr = dev_.default_queue();
  dev_.queues_wait_and_throw();
  QueuePtr = dev_.create_queue();
  dev_.destroy_queue(QueuePtr);
  QueuePtr = dev_.create_queue();
  dev_.set_saved_queue(QueuePtr);
  QueuePtr = dev_.get_saved_queue();
  auto Context = dev_.get_context();
}

TEST_F(DeviceExtFixt, DefaultSavedQueue) {
  EXPECT_EQ(*dev_.default_queue(), *dev_.get_saved_queue());
}

TEST_F(DeviceExtFixt, SavedQueue) {
  auto q = *dev_.create_queue();
  dev_.set_saved_queue(&q);
  EXPECT_EQ(q, *dev_.get_saved_queue());
}

// Check reset() resets the queues etc
TEST_F(DeviceExtFixt, Reset) {
  auto q = *dev_.create_queue();
  dev_.set_saved_queue(&q);
  dev_.reset();
  EXPECT_NE(q, *dev_.get_saved_queue());
}

TEST(DeviceInfo, DeviceInfoAPI) {
  syclcompat::device_info Info;
  const char *Name = "DEVNAME";
  Info.set_name(Name);
  sycl::id<3> max_work_item_sizes;
  Info.set_max_work_item_sizes(max_work_item_sizes);
  Info.set_major_version(1);
  Info.set_minor_version(1);
  Info.set_integrated(1);
  Info.set_max_clock_frequency(1000);
  Info.set_max_compute_units(32);
  Info.set_global_mem_size(1000);
  Info.set_local_mem_size(1000);
  Info.set_max_work_group_size(32);
  Info.set_max_sub_group_size(16);
  Info.set_max_work_items_per_compute_unit(16);
  int SizeArray[3] = {1, 2, 3};
  Info.set_max_nd_range_size(SizeArray);

  EXPECT_STREQ(Info.get_name(), Name);
  EXPECT_EQ(Info.get_max_work_item_sizes(), max_work_item_sizes);
  EXPECT_EQ(Info.get_minor_version(), 1);
  EXPECT_EQ(Info.get_integrated(), 1);
  EXPECT_EQ(Info.get_max_clock_frequency(), 1000);
  EXPECT_EQ(Info.get_max_compute_units(), 32);
  EXPECT_EQ(Info.get_max_work_group_size(), 32);
  EXPECT_EQ(Info.get_max_sub_group_size(), 16);
  EXPECT_EQ(Info.get_max_work_items_per_compute_unit(), 16);
  EXPECT_EQ(Info.get_max_nd_range_size()[0], SizeArray[0]);
  EXPECT_EQ(Info.get_max_nd_range_size()[1], SizeArray[1]);
  EXPECT_EQ(Info.get_max_nd_range_size()[2], SizeArray[2]);
  EXPECT_EQ(Info.get_global_mem_size(), 1000);
  EXPECT_EQ(Info.get_local_mem_size(), 1000);
}

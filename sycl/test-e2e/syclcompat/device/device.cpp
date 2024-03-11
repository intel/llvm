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
 *  device.cpp
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

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/device.hpp>

#include "device_fixt.hpp"

void test_set_default_queue() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();
  sycl::queue old_default_queue = syclcompat::get_default_queue();
  dev_.set_default_queue(syclcompat::create_queue());
  assert(*dev_.default_queue() == *dev_.get_saved_queue());
  assert(*dev_.default_queue() != old_default_queue);
}

int main() {
  /*
    Device Tests
  */
  std::cout << "Testing AtLeastOneDevice" << std::endl;
  {
    DeviceTestsFixt dtf;
    assert(dtf.get_n_devices() > 0);
  }

  // Check the device returned matches the device ID
  std::cout << "Testing MatchesID" << std::endl;
  {
    assert(syclcompat::get_device(syclcompat::get_current_device_id()) ==
           syclcompat::get_current_device());
  }

  // Check error on insufficient devices
  std::cout << "Testing NotEnoughDevices" << std::endl;
  {
    DeviceTestsFixt dtf;
    try {
      syclcompat::select_device(dtf.get_n_devices());
    } catch (std::runtime_error const &e) {
      std::cout << "Expected SYCL exception caught: " << e.what();
    }
  }

  // Check the default context matches default queue's context
  std::cout << "Testing DefaultContext" << std::endl;
  {
    DeviceTestsFixt dtf;
    assert(dtf.get_queue().get_context() == syclcompat::get_default_context());
  }

  /*
    Queue Tests
  */
  std::cout << "Testing MakeInOrderQueue" << std::endl;
  {
    sycl::queue q = syclcompat::get_default_queue();
    assert(q.is_in_order());
  }

  std::cout << "Testing CheckDefaultDevice" << std::endl;
  {
    sycl::queue q = syclcompat::get_default_queue();
    assert(q.get_device() == sycl::device{sycl::default_selector_v});
  }

  // Check behaviour of in order & out of order queue construction
  std::cout << "Testing QueuePropOrder" << std::endl;
  {
    sycl::queue q_create_def{syclcompat::create_queue()};
    assert(q_create_def.is_in_order());
    sycl::queue q_in_order{syclcompat::create_queue(false, true)};
    assert(q_in_order.is_in_order());
    sycl::queue q_out_order{syclcompat::create_queue(false, false)};
    assert(!q_out_order.is_in_order());
  }

  /*
    Device Extension Tests
  */
  std::cout << "Testing DeviceExtAPI" << std::endl;
  {
    DeviceExtFixt dev_ext;
    auto &dev_ = dev_ext.get_dev_ext();
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

  std::cout << "Testing DefaultSavedQueue" << std::endl;
  {
    DeviceExtFixt dev_ext;
    auto &dev_ = dev_ext.get_dev_ext();
    assert(*dev_.default_queue() == *dev_.get_saved_queue());
  }

  std::cout << "Testing SavedQueue" << std::endl;
  {
    DeviceExtFixt dev_ext;
    auto &dev_ = dev_ext.get_dev_ext();
    auto q = *dev_.create_queue();
    dev_.set_saved_queue(&q);
    assert(q == *dev_.get_saved_queue());
  }

  // Check reset() resets the queues etc
  std::cout << "Testing Reset" << std::endl;
  {
    DeviceExtFixt dev_ext;
    auto &dev_ = dev_ext.get_dev_ext();
    auto q = *dev_.create_queue();
    dev_.set_saved_queue(&q);
    dev_.reset();
    assert(q != *dev_.get_saved_queue());
  }

  std::cout << "Testing DeviceInfoAPI" << std::endl;
  {
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

    assert(!strcmp(Info.get_name(), Name));
    assert(Info.get_max_work_item_sizes() == max_work_item_sizes);
    assert(Info.get_minor_version() == 1);
    assert(Info.get_integrated() == 1);
    assert(Info.get_max_clock_frequency() == 1000);
    assert(Info.get_max_compute_units() == 32);
    assert(Info.get_max_work_group_size() == 32);
    assert(Info.get_max_sub_group_size() == 16);
    assert(Info.get_max_work_items_per_compute_unit() == 16);
    assert(Info.get_max_nd_range_size()[0] == SizeArray[0]);
    assert(Info.get_max_nd_range_size()[1] == SizeArray[1]);
    assert(Info.get_max_nd_range_size()[2] == SizeArray[2]);
    assert(Info.get_global_mem_size() == 1000);
    assert(Info.get_local_mem_size() == 1000);
  }

  return 0;
}

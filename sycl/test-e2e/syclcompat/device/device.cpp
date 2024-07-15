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

/*
  Device Tests
*/
void test_at_least_one_device() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceTestsFixt dtf;
  assert(dtf.get_n_devices() > 0);
}

// Check the device returned matches the device ID
void test_matches_id() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  assert(syclcompat::get_device(syclcompat::get_current_device_id()) ==
         syclcompat::get_current_device());
}

// Check error on insufficient devices
void test_not_enough_devices() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceTestsFixt dtf;
  try {
    syclcompat::select_device(dtf.get_n_devices());
  } catch (std::runtime_error const &e) {
    std::cout << "Expected SYCL exception caught: " << e.what() << std::endl;
  }
}

// Check the default context matches default queue's context
void test_default_context() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceTestsFixt dtf;
  assert(dtf.get_queue().get_context() == syclcompat::get_default_context());
}

/*
  Queue Tests
*/
void test_make_in_order_queue() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q = syclcompat::get_default_queue();
  assert(q.is_in_order());
}

void test_check_default_device() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q = syclcompat::get_default_queue();
  assert(q.get_device() == sycl::device{sycl::default_selector_v});
}

// Check behaviour of in order & out of order queue construction
void test_create_queue_arguments() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q_create_def{syclcompat::create_queue()};
  assert(q_create_def.is_in_order());
  sycl::queue q_in_order{syclcompat::create_queue(false, true)};
  assert(q_in_order.is_in_order());
  sycl::queue q_out_order{syclcompat::create_queue(false, false)};
  assert(!q_out_order.is_in_order());
}

void test_version_parsing_case(const std::string &ver_string,
                               int expected_major, int expected_minor) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  int major;
  int minor;
  syclcompat::detail::parse_version_string(ver_string, major, minor);
  if (major != expected_major || minor != expected_minor) {
    std::cout << "Failed comparing " << ver_string << " major " << major
              << " expected_major " << expected_major << " minor " << minor
              << " expected_minor " << expected_minor << std::endl;
    assert(false);
  }
  assert(major == expected_major);
  assert(minor == expected_minor);
}

void test_version_parsing() {
  test_version_parsing_case("3.0", 3, 0);
  test_version_parsing_case("3.0 NEO", 3, 0);
  test_version_parsing_case("OpenCL 3.0 NEO", 3, 0);
  test_version_parsing_case("OpenCL 3.0 (Build 0)", 3, 0);
  test_version_parsing_case("8.6", 8, 6);
  test_version_parsing_case("8.0", 8, 0);
  test_version_parsing_case("7.5", 7, 5);
  test_version_parsing_case("1.3", 1, 3);
  test_version_parsing_case("11.4", 11, 4);
  test_version_parsing_case("0.1", 0, 1);
  test_version_parsing_case("gfx1030", 1030, 0);
}

// We have *some* constraints on the major version that we can check
void test_major_version(sycl::device &dev, int major) {
  auto backend = dev.get_backend();
  if (backend == sycl::backend::opencl) {
    assert(major == 1 || major == 3);
  } else if (backend == sycl::backend::ext_oneapi_level_zero ||
             backend == sycl::backend::ext_oneapi_cuda) {
    assert(major < 99);
  }
}

/*
  Device Extension Tests
*/
void test_device_ext_api() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();
  dev_.is_native_host_atomic_supported();
  auto major = dev_.get_major_version();
  test_major_version(dev_, major);
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

void test_device_api() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();
  auto major = get_major_version(dev_);
  test_major_version(dev_, major);
  get_minor_version(dev_);
}

void test_default_saved_queue() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();
  assert(*dev_.default_queue() == *dev_.get_saved_queue());
}

void test_saved_queue() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();
  auto q = *dev_.create_queue();
  dev_.set_saved_queue(&q);
  assert(q == *dev_.get_saved_queue());
}

// Check reset() resets the queues etc
void test_reset() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();
  auto q = *dev_.create_queue();
  dev_.set_saved_queue(&q);
  dev_.reset();
  assert(q != *dev_.get_saved_queue());
}

void test_reset_arguments() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();

  dev_.reset(false, false);
  assert(!dev_.get_saved_queue()->is_in_order());

  dev_.reset(false, true);
  assert(dev_.get_saved_queue()->is_in_order());
}

void test_device_info_api() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  syclcompat::device_info Info;
  const char *Name = "DEVNAME";
  std::array<unsigned char, 16> uuid;
  uuid.fill('0');
  sycl::range<3> max_work_item_sizes;

  Info.set_name(Name);
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

  Info.set_host_unified_memory(true);
  Info.set_memory_clock_rate(1000);
  Info.set_max_register_size_per_work_group(1000);
  Info.set_device_id(0);
  Info.set_uuid(uuid);
  Info.set_global_mem_cache_size(1000);

  assert(!strcmp(Info.get_name(), Name));
  assert(Info.get_max_work_item_sizes() == max_work_item_sizes);
  assert(Info.get_minor_version() == 1);
  assert(Info.get_integrated() == 1);
  assert(Info.get_max_clock_frequency() == 1000);
  assert(Info.get_max_compute_units() == 32);
  assert(Info.get_max_work_group_size() == 32);
  assert(Info.get_max_sub_group_size() == 16);
  assert(Info.get_max_work_items_per_compute_unit() == 16);
  assert(Info.get_global_mem_size() == 1000);
  assert(Info.get_local_mem_size() == 1000);

  uuid.fill('0'); // set_uuid uses std::move
  assert(Info.get_host_unified_memory());
  assert(Info.get_memory_clock_rate() == 1000);
  assert(Info.get_max_register_size_per_work_group() == 1000);
  assert(Info.get_device_id() == 0);
  assert(Info.get_uuid() == uuid);
  assert(Info.get_global_mem_cache_size() == 1000);
}

void test_image_max_attrs() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  syclcompat::device_info info;

  int _image1d_max = 1;
  int _image2d_max[2] = {2, 3};
  int _image3d_max[3] = {4, 5, 6};

  info.set_image1d_max(_image1d_max);
  info.set_image2d_max(_image2d_max[0], _image2d_max[1]);
  info.set_image3d_max(_image3d_max[0], _image3d_max[1], _image3d_max[2]);

  assert(info.get_image1d_max() == _image1d_max);
  assert(info.get_image2d_max()[0] == _image2d_max[0]);
  assert(info.get_image2d_max()[1] == _image2d_max[1]);
  assert(info.get_image3d_max()[0] == _image3d_max[0]);
  assert(info.get_image3d_max()[1] == _image3d_max[1]);
  assert(info.get_image3d_max()[2] == _image3d_max[2]);

  DeviceExtFixt dev_ext;
  auto &dev_ = dev_ext.get_dev_ext();

  info.set_image1d_max(0);
  info.set_image2d_max(0, 0);
  info.set_image3d_max(0, 0, 0);

  // SYCL guarantees at least a certain minimum value if the device has
  // aspect::image
  if (!dev_.has(sycl::aspect::image)) {
    std::cout << "  Partial skip: device does not have sycl::aspect::image."
              << std::endl;
    return;
  }
  dev_.get_device_info(info);
  // We only need to ensure the value is modified.
  assert(info.get_image1d_max() > 0);
  assert(info.get_image2d_max()[0] > 0);
  assert(info.get_image2d_max()[1] > 0);
  assert(info.get_image3d_max()[0] > 0);
  assert(info.get_image3d_max()[1] > 0);
  assert(info.get_image3d_max()[2] > 0);
}

void test_max_nd_range() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  syclcompat::device_info info;

  int size_array[3] = {1, 2, 3};
  info.set_max_nd_range_size(size_array);

  assert(info.get_max_nd_range_size()[0] == size_array[0]);
  assert(info.get_max_nd_range_size()[1] == size_array[1]);
  assert(info.get_max_nd_range_size()[2] == size_array[2]);

  DeviceExtFixt dev_ext;
  auto &dev = dev_ext.get_dev_ext();
  dev.get_device_info(info);

  int size_array_zeros[3] = {0, 0, 0};
  info.set_max_nd_range_size(size_array_zeros);

#ifdef SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY
  // According to the extension values are > 1 unless info::device_type is
  // info::device_type::custom.
  if (dev.get_info<sycl::info::device::device_type>() ==
      sycl::info::device_type::custom) {
    std::cout << "  Skipping due to custom sycl::info::device_type::custom."
              << std::endl;
    return;
  }

  info.set_max_nd_range_size(
      dev.get_info<
          sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>());
  assert(info.get_max_nd_range_size()[0] > 0);
  assert(info.get_max_nd_range_size()[1] > 0);
  assert(info.get_max_nd_range_size()[2] > 0);
#else
  int expected = 0x7FFFFFFF;
  assert(info.get_max_nd_range_size()[0] == expected);
  assert(info.get_max_nd_range_size()[1] == expected);
  assert(info.get_max_nd_range_size()[2] == expected);
#endif
}

void test_list_devices() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  DeviceTestsFixt dtf;

  // Redirect std::cout to count new lines
  CountingStream countingBuf(std::cout.rdbuf());
  std::streambuf *orig_buf = std::cout.rdbuf();
  std::cout.rdbuf(&countingBuf);

  syclcompat::list_devices();

  // Restore back std::cout
  std::cout.rdbuf(orig_buf);

  // Expected one line per device
  assert(countingBuf.get_line_count() == dtf.get_n_devices());
}

void test_device_count() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  unsigned int count = syclcompat::device_count();
  assert(count > 0);
}

void test_get_device_id() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::device dev = syclcompat::get_device(0);
  unsigned int id = syclcompat::get_device_id(dev);
  assert(id == 0);
}

int main() {
  test_at_least_one_device();
  test_matches_id();
  test_not_enough_devices();
  test_set_default_queue();
  test_default_context();
  test_make_in_order_queue();
  test_check_default_device();
  test_create_queue_arguments();
  test_device_ext_api();
  test_device_api();
  test_default_saved_queue();
  test_saved_queue();
  test_reset();
  test_device_info_api();
  test_version_parsing();
  test_image_max_attrs();
  test_max_nd_range();
  test_list_devices();
  test_device_count();
  test_get_device_id();

  return 0;
}

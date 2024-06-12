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
 *  device_filter.cpp
 *
 *  Description:
 *    Device filtering tests
 **************************************************************************/

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/device.hpp>

void test_filtering_existing_device() {
  auto &dev = syclcompat::get_current_device();
  std::string dev_name = dev.get_info<sycl::info::device::name>();

  syclcompat::filter_device({dev_name});
  try {
    syclcompat::get_device_id(dev);
  } catch (std::runtime_error const &e) {
    std::cout << "  Unexpected SYCL exception caught: " << e.what()
              << std::endl;
    assert(0);
  }

  // Checks for a substring of the device as well
  std::string dev_substr = dev_name.substr(1, dev_name.find(" ") + 2);
  syclcompat::filter_device({dev_substr});
  try {
    syclcompat::get_device_id(dev);
  } catch (std::runtime_error const &e) {
    std::cout << "  Unexpected SYCL exception caught: " << e.what()
              << std::endl;
    assert(0);
  }
}

void test_filter_devices() {
  auto &dev = syclcompat::get_current_device();

  assert(syclcompat::detail::dev_mgr::instance().device_count() > 0);

  syclcompat::filter_device({"NON-EXISTENT DEVICE"});
  assert(syclcompat::detail::dev_mgr::instance().device_count() == 0);

  try {
    syclcompat::get_device_id(dev);
    assert(0);
  } catch (std::runtime_error const &e) {
    std::cout << "  Expected SYCL exception caught: " << e.what() << std::endl;
  }
}

int main() {
  // syclcompat::dev_mgr is a singleton, so any changes to the device list is
  // permanent between tests. Test isolated instead of relying on it being the
  // last test in a different test suite.
  test_filtering_existing_device();

  test_filter_devices();

  return 0;
}

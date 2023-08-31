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
 *  atomic_fixt.hpp
 *
 *  Description:
 *    Memory Order helper for the Atomic functionality tests
 **************************************************************************/

#pragma once

#include <algorithm>
#include <sycl/sycl.hpp>

#include <syclcompat.hpp>

using atomic_value_type_list =
    std::tuple<int, unsigned int, long, unsigned long, long long,
               unsigned long long, float, double>;

using atomic_ptr_type_list =
    std::tuple<int *, long *, long long *, float *, double *>;

using integral_type_list = std::tuple<int, unsigned int, long, unsigned long,
                                      long long, unsigned long long>;

using signed_type_list = std::tuple<int, long, long long, float, double>;

bool is_supported(std::vector<sycl::memory_order> capabilities,
                  sycl::memory_order mem_order) {
  return std::find(capabilities.begin(), capabilities.end(), mem_order) !=
         capabilities.end();
}

template <auto F, typename T> class AtomicLauncher {
protected:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  T *data_;
  sycl::queue q_;
  bool skip_;

  bool should_skip() {
    if (!q_.get_device().has(sycl::aspect::fp64) &&
        (std::is_same_v<T, double> || std::is_same_v<T, double *>))
      return true;
    return false;
  }

public:
  AtomicLauncher(syclcompat::dim3 grid, syclcompat::dim3 threads,
                 sycl::queue q = syclcompat::get_default_queue())
      : grid_{grid}, threads_{threads}, q_{q}, skip_{should_skip()} {
    data_ = (T *)syclcompat::malloc(sizeof(T));
  };
  ~AtomicLauncher() { syclcompat::free(data_); }
  template <typename... Args>
  void launch_test(T init_val, T expected_result, Args... args) {
    if (skip_)
      return;

    syclcompat::memcpy(data_, &init_val, sizeof(T));
    syclcompat::launch<F>(grid_, threads_, data_, args...);
    T result_val;
    syclcompat::memcpy(&result_val, data_, sizeof(T));
    syclcompat::wait();
    assert(result_val == expected_result);
  }
};

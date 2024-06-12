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
#include <sycl/detail/core.hpp>

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

template <typename T> bool should_skip(const sycl::device &dev) {
  if constexpr (sizeof(T) == 8) {
    if (!dev.has(sycl::aspect::atomic64)) {
      return true;
    }
  }
  if constexpr (std::is_same_v<T, double> || std::is_same_v<T, double *>) {
    if (!dev.has(sycl::aspect::fp64)) {
      return true;
    }
  }
  return false;
}

template <auto F, typename T> class AtomicLauncher {
protected:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  T *data_;
  sycl::queue q_;
  bool skip_;

public:
  AtomicLauncher(syclcompat::dim3 grid, syclcompat::dim3 threads,
                 sycl::queue q = syclcompat::get_default_queue())
      : grid_{grid}, threads_{threads}, q_{q},
        skip_{should_skip<T>(q.get_device())} {
    data_ = (T *)syclcompat::malloc(sizeof(T), q_);
  };
  ~AtomicLauncher() { syclcompat::free(data_); }
  template <typename... Args>
  void launch_test(T init_val, T expected_result, Args... args) {
    if (skip_)
      return;

    syclcompat::memcpy(data_, &init_val, sizeof(T), q_);
    syclcompat::launch<F>(grid_, threads_, q_, data_, args...);
    T result_val;
    syclcompat::memcpy(&result_val, data_, sizeof(T), q_);
    syclcompat::wait();
    assert(result_val == expected_result);
  }
};

template <typename T> class AtomicClassLauncher {
protected:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  size_t data_len_;
  T *atom_arr_device_;
  T *atom_arr_host_;
  bool skip_;

  void verify() {
    bool result = true;
    for (int i = 0; i < data_len_; ++i) {
      if (atom_arr_device_[i] != atom_arr_host_[i]) {
        std::cout << "-- Failure at " << i << std::endl << std::flush;
        result = false;
      }
    }
    assert(result);
  }

public:
  AtomicClassLauncher(const syclcompat::dim3 &grid,
                      const syclcompat::dim3 &threads, const size_t data_len)
      : grid_{grid}, threads_{threads}, data_len_{data_len},
        skip_{should_skip<T>(syclcompat::get_current_device())} {
    atom_arr_device_ = syclcompat::malloc_shared<T>(data_len_);
    atom_arr_host_ = syclcompat::malloc_shared<T>(data_len_);

    for (size_t i = 0; i < data_len_; i++) {
      atom_arr_device_[i] = 0;
      atom_arr_host_[i] = 0;
    }
  };
  virtual ~AtomicClassLauncher() {
    syclcompat::free(atom_arr_device_);
    syclcompat::free(atom_arr_host_);
  }

  template <auto Kernel, auto HostFunc> void launch_test() {
    if (skip_)
      return; // skip
    syclcompat::launch<Kernel>(grid_, threads_, atom_arr_device_);
    HostFunc(atom_arr_host_);
    syclcompat::wait();

    verify();
  }
};

template <typename T>
class AtomicClassPtrTypeLauncher : public AtomicClassLauncher<T> {
protected:
  using ValType = std::remove_pointer_t<T>;

  T *atom_arr_shared_in_;

public:
  AtomicClassPtrTypeLauncher(const syclcompat::dim3 &grid,
                             const syclcompat::dim3 &threads,
                             const size_t data_len)
      : AtomicClassLauncher<T>(grid, threads, data_len) {

    atom_arr_shared_in_ = syclcompat::malloc_shared<T>(this->data_len_);

    for (size_t i = 0; i < this->data_len_; i++) {
      atom_arr_shared_in_[i] = syclcompat::malloc_shared<ValType>(1);
    }
  };

  virtual ~AtomicClassPtrTypeLauncher() {
    for (size_t i = 0; i < this->data_len_; i++) {
      syclcompat::free(atom_arr_shared_in_[i]);
    }
    syclcompat::free(atom_arr_shared_in_);
  }

  template <auto Kernel, auto HostFunc> void launch_test() {
    if (this->skip_)
      return;
    syclcompat::launch<Kernel>(this->grid_, this->threads_,
                               this->atom_arr_device_, atom_arr_shared_in_);
    HostFunc(this->atom_arr_host_, atom_arr_shared_in_);
    syclcompat::wait();

    this->verify();
  }
};

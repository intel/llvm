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
 *  SYCLcompat
 *
 *  id_query_fixt.hpp
 *
 *  Description:
 *     Fixtures and helpers for to tests the id_query functionality
 **************************************************************************/

#pragma once

#include <sycl/detail/core.hpp>

#include <syclcompat/id_query.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

// Class to launch a kernel and run a lambda on output data
template <auto F> class QueryLauncher {
protected:
  syclcompat::dim3 grid_;
  syclcompat::dim3 threads_;
  size_t size_;
  int *data_;
  std::vector<int> host_data_;
  using CheckLambda = std::function<void(std::vector<int>)>;

public:
  QueryLauncher(syclcompat::dim3 grid, syclcompat::dim3 threads)
      : grid_{grid}, threads_{threads}, size_{grid_.size() * threads_.size()},
        host_data_(size_) {
    data_ = (int *)syclcompat::malloc(size_ * sizeof(int));
    syclcompat::memset(data_, 0, size_ * sizeof(int));
  };
  ~QueryLauncher() { syclcompat::free(data_); }
  template <typename... Args>
  void launch_dim3(CheckLambda checker, Args... args) {
    syclcompat::launch<F>(grid_, threads_, data_, args...);
    syclcompat::memcpy(host_data_.data(), data_, size_ * sizeof(int));
    syclcompat::wait();
    checker(host_data_);
  }
  template <int Dim, typename... Args>
  void launch_ndrange(CheckLambda checker, Args... args) {
    sycl::nd_range<Dim> range = {grid_ * threads_, grid_};
    syclcompat::launch<F>(range, data_, args...);
    syclcompat::memcpy(host_data_.data(), data_, size_ * sizeof(int));
    syclcompat::wait();
    checker(host_data_);
  }
};

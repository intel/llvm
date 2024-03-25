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
 *  math_length_test.cpp
 *
 *  Description:
 *    vector length tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilFastLengthTest.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <numeric>

#include <sycl/detail/core.hpp>

#include <syclcompat.hpp>

#define MAX_LEN 5

void compute_fast_length(float *d_A, size_t n, float *ans) {
  *ans = syclcompat::fast_length(d_A, n);
}

void compute_length(float *d_A, size_t n, float *ans) {
  *ans = syclcompat::length(d_A, n);
}

class LengthLauncher {
protected:
  float *data_;
  float *result_;
  float host_result_{0.0};

public:
  LengthLauncher() {
    data_ = (float *)syclcompat::malloc(MAX_LEN * sizeof(float));
    result_ = (float *)syclcompat::malloc(sizeof(float));
  };
  ~LengthLauncher() {
    syclcompat::free(data_);
    syclcompat::free(result_);
  }

  void check_result(std::vector<float> result) {
    float sum =
        std::inner_product(result.begin(), result.end(), result.begin(), 0.0f);
    float diff = fabs(sqrtf(sum)) - host_result_;
    assert(diff <= 1.e-5);
  }

  template <auto F> void launch(std::vector<float> vec) {
    size_t n = vec.size();
    syclcompat::memcpy(data_, vec.data(), sizeof(float) * n);
    auto data = data_;
    auto result = result_;
    syclcompat::get_default_queue().single_task(
        [data, result, n]() { F(data, n, result); });
    syclcompat::memcpy(&host_result_, result_, sizeof(float));
    check_result(vec);
  }
};

void test_fast_length() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  auto launcher = LengthLauncher();
  launcher.launch<compute_fast_length>(std::vector<float>{0.8970062715});
  launcher.launch<compute_fast_length>(
      std::vector<float>{0.8335529744, 0.7346600673});
  launcher.launch<compute_fast_length>(
      std::vector<float>{0.1658983906, 0.590226484, 0.4891553616});
  launcher.launch<compute_fast_length>(std::vector<float>{
      0.6041178723, 0.7760620605, 0.2944284976, 0.6851913766});
  launcher.launch<compute_fast_length>(std::vector<float>{
      0.6041178723, 0.7760620605, 0.2944284976, 0.6851913766, 0.6851913766});
}

void test_length() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  auto launcher = LengthLauncher();
  launcher.launch<compute_length>(std::vector<float>{0.8970062715});
  launcher.launch<compute_length>(
      std::vector<float>{0.8335529744, 0.7346600673});
  launcher.launch<compute_length>(
      std::vector<float>{0.1658983906, 0.590226484, 0.4891553616});
  launcher.launch<compute_length>(std::vector<float>{
      0.6041178723, 0.7760620605, 0.2944284976, 0.6851913766});
  launcher.launch<compute_length>(std::vector<float>{
      0.6041178723, 0.7760620605, 0.2944284976, 0.6851913766, 0.6851913766});
}

int main() {
  test_fast_length();
  test_length();

  return 0;
}

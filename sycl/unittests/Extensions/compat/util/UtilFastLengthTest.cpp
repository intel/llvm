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
 *  SYCL compatibility API
 *
 *  UtilFastLengthTest.cpp
 *
 *  Description:
 *    Fast length tests
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

#include <gtest/gtest.h>
#include <numeric>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

#define MAX_LEN 5

void compute_length(float *d_A, size_t n, float *ans) {
  *ans = sycl::ext::oneapi::experimental::compat::fast_length(d_A, n);
}

class FastLengthLauncher {
protected:
  float *data_;
  float *result_;
  float host_result_{0.0};

public:
  FastLengthLauncher() {
    data_ = (float *)sycl::ext::oneapi::experimental::compat::malloc(
        MAX_LEN * sizeof(float));
    result_ =
        (float *)sycl::ext::oneapi::experimental::compat::malloc(sizeof(float));
  };
  ~FastLengthLauncher() {
    sycl::ext::oneapi::experimental::compat::free(data_);
    sycl::ext::oneapi::experimental::compat::free(result_);
  }

  void check_result(std::vector<float> result) {
    float sum =
        std::inner_product(result.begin(), result.end(), result.begin(), 0.0f);
    float diff = fabs(sqrtf(sum)) - host_result_;
    EXPECT_LE(diff, 1.e-5);
  }

  void launch(std::vector<float> vec) {
    size_t n = vec.size();
    sycl::ext::oneapi::experimental::compat::memcpy(data_, vec.data(),
                                                    sizeof(float) * n);
    auto data = data_;
    auto result = result_;
    sycl::ext::oneapi::experimental::compat::get_default_queue().single_task(
        [data, result, n]() { compute_length(data, n, result); });
    sycl::ext::oneapi::experimental::compat::memcpy(&host_result_, result_,
                                                    sizeof(float));
    check_result(vec);
  }
};

TEST(Util, fast_length) {
  auto launcher = FastLengthLauncher();
  launcher.launch(std::vector<float>{0.8970062715});
  launcher.launch(std::vector<float>{0.8335529744, 0.7346600673});
  launcher.launch(std::vector<float>{0.1658983906, 0.590226484, 0.4891553616});
  launcher.launch(std::vector<float>{0.6041178723, 0.7760620605, 0.2944284976,
                                     0.6851913766});
  launcher.launch(std::vector<float>{0.6041178723, 0.7760620605, 0.2944284976,
                                     0.6851913766, 0.6851913766});
}

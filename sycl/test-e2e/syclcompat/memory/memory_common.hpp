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
 *  memory_common.hpp
 *
 *  Description:
 *    Memory content helper for the Memory functionality tests
 **************************************************************************/

#pragma once

#include <cassert>
#include <cmath>
#include <tuple>

#include <sycl/detail/core.hpp>

inline void check(float *h_data, float *h_ref, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float diff = fabs(h_data[i] - h_ref[i]);
    assert(diff <= 1.e-6);
  }
}

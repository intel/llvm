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
 *  UtilReverseBitsTest.cpp
 *
 *  Description:
 *    reverse_bits tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilReverseBitsTest.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

TEST(Util, reverse_bits) {
  unsigned int a = 1;
  unsigned int b = sycl::ext::oneapi::experimental::compat::reverse_bits(a);
  EXPECT_EQ(b, 0x80000000);

  a = 0x12345678;
  b = sycl::ext::oneapi::experimental::compat::reverse_bits(a);
  EXPECT_EQ(b, 0x1e6a2c48);
}

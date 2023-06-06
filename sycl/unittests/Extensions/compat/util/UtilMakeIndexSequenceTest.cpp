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
 *  UtilMakeIndexSequenceTest.cpp
 *
 *  Description:
 *    make_index_sequence tests
 **************************************************************************/

// The original source was under the license below:
// ====------ UtilMakeIndexSequenceTest.cpp---------- -*- C++ -* ----===////
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

using namespace sycl::ext::oneapi::experimental;

const int ref_range[3] = {1, 2, 3};

template <int... DimIdx>
sycl::range<sizeof...(DimIdx)>
get_range(compat::detail::integer_sequence<DimIdx...>) {
  return sycl::range<sizeof...(DimIdx)>(ref_range[DimIdx]...);
}

TEST(Util, make_index_sequence) {
  const int dimensions = 3;
  auto index = compat::detail::make_index_sequence<dimensions>();

  auto value = get_range(index);

  for (int i = 0; i < dimensions; i++) {
    EXPECT_EQ(value[i], ref_range[i]);
  }
}

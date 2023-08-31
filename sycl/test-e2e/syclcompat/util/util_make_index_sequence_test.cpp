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
 *  util_make_index_sequence_test.cpp
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

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

const int ref_range[3] = {1, 2, 3};

template <int... DimIdx>
sycl::range<sizeof...(DimIdx)>
get_range(syclcompat::detail::integer_sequence<DimIdx...>) {
  return sycl::range<sizeof...(DimIdx)>(ref_range[DimIdx]...);
}

void test_make_index_sequence() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const int dimensions = 3;
  auto index = syclcompat::detail::make_index_sequence<dimensions>();

  auto value = get_range(index);

  for (int i = 0; i < dimensions; i++) {
    assert(value[i] == ref_range[i]);
  }
}

int main() {
  test_make_index_sequence();

  return 0;
}

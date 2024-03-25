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
 *  math_complex_datatype.cpp
 *
 *  Description:
 *    Complex operations tests
 **************************************************************************/

// The original source was under the license below:
//===-------------- UtilComplex.cpp --------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <syclcompat/math.hpp>
#include <syclcompat/util.hpp>

void test_datatype() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  if constexpr (!std::is_same<syclcompat::detail::DataType<float>::T2,
                              float>::value)
    assert(false); // FAIL
#ifdef SYCL_EXT_ONEAPI_COMPLEX
  if constexpr (!std::is_same<
                    syclcompat::detail::DataType<sycl::float2>::T2,
                    sycl::ext::oneapi::experimental::complex<float>>::value)
    assert(false); // FAIL
#endif
}

int main() {
  test_datatype();

  return 0;
}

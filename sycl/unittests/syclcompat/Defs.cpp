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
 *  Defs.cpp
 *
 *  Description:
 *     __sycl_compat_align__ tests
 **************************************************************************/

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <syclcompat/defs.hpp>

TEST(DEFS, Align) {
  struct __sycl_compat_align__(16) {
    int a;
    char c;
  } s;
  EXPECT_EQ(sizeof(s), 16);
}

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

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <syclcompat/defs.hpp>

int main() {
  struct __sycl_compat_align__(16) {
    int a;
    char c;
  }
  s;
  assert(sizeof(s) == 16);

  return 0;
}

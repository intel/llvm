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
 *  memory_adl.cpp
 *
 *  Description:
 *    Tests to ensure global namespace functions don't clash via ADL
 **************************************************************************/

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only
// Test that no syclcompat:: functions clash with global namespace fns due to ADL
#include <sycl/sycl.hpp>
#include <syclcompat/syclcompat.hpp>

int main(){
  syclcompat::device_info dummy_info;
  syclcompat::device_info dummy_info_2;
  memset(&dummy_info, 0, sizeof(syclcompat::device_info));
  memcpy(&dummy_info, &dummy_info_2, sizeof(syclcompat::device_info));
  free(&dummy_info);
}

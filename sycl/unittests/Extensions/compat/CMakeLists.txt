#
# Copyright (C) Codeplay Software Ltd.
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SYCL compatibility API
#

set(SYCL_COMPAT_TEST_FLAGS -fsycl -fsycl-unnamed-lambda
                           -fsycl-device-code-split=per_kernel)

macro(add_sycl_unittest_with_extra_flags test_dirname link_variant)
  add_sycl_unittest(${test_dirname} ${link_variant} ${ARGN})

  target_compile_options(${test_dirname} PRIVATE ${SYCL_COMPAT_TEST_FLAGS})
  target_link_options(${test_dirname} PRIVATE ${SYCL_COMPAT_TEST_FLAGS})
endmacro()

add_sycl_unittest_with_extra_flags(Extensionscompattests SHARED
  Device.cpp
  Dim.cpp
  Defs.cpp
  IdQuery.cpp
  Launch.cpp
)

add_subdirectory(atomic)
add_subdirectory(kernel)
add_subdirectory(memory)
add_subdirectory(util)

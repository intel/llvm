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
# SYCLcompat
#

set(SYCL_EXTRA_FLAGS "-fsycl-unnamed-lambda;-fsycl-device-code-split=per_kernel")

if (UNIX)
  add_custom_target(check-syclcompat)
  add_dependencies(check-sycl-unittests check-syclcompat)
endif()

add_sycl_library_unittest(syclcompat-base
  SYCL_EXTRA_FLAGS
    ${SYCL_EXTRA_FLAGS}
  SOURCES
    Dim.cpp
    Defs.cpp)

if (UNIX)
  add_dependencies(check-syclcompat check-syclcompat-base)
endif()

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

add_sycl_unittest_with_extra_flags(Extensionscompatatomictests SHARED 
  Atomic.cpp
  AtomicMemoryOrderAcqRel.cpp
)

target_include_directories(Extensionscompatatomictests 
                           PRIVATE ${CURRENT_DIR})
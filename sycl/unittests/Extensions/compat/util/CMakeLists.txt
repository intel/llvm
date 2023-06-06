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

add_sycl_unittest_with_extra_flags(Extensionscompatutiltests SHARED
  UtilCastValueTest.cpp
  UtilComplex.cpp
  UtilFastLengthTest.cpp
  UtilFindFirstSet.cpp
  UtilLogicalGroup.cpp
  UtilMakeIndexSequenceTest.cpp
  UtilMatrixMemCopyTest.cpp
  UtilNdRangeBarrierTest.cpp
  UtilPermByteTest.cpp
  UtilPermuteSubGroupByXor.cpp
  UtilReverseBitsTest.cpp
  UtilSelectFromSubGroup.cpp
  UtilShiftSubGroupLeft.cpp
  UtilShiftSubGroupRight.cpp
  UtilVectorizedIsgreaterTest.cpp
  UtilVectorizedMaxTest.cpp
  UtilVectorizedMinTest.cpp
)

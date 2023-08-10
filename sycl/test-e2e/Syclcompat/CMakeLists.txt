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
if(SYCL_TEST_E2E_STANDALONE)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py")
else()
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
    )
endif() # Standalone.

add_custom_target(check-sycl-e2e-syclcompat
  COMMAND ${Python3_EXECUTABLE} ${LLVM_LIT} ${SYCL_E2E_TESTS_LIT_FLAGS} .
  COMMENT "Running SYCLcompat End-to-End tests"
  DEPENDS ${SYCL_E2E_TEST_DEPS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  USES_TERMINAL
)
add_dependencies(check-sycl-e2e check-sycl-e2e-syclcompat)

include(AddSYCLLibraryUnitTest)
set(SYCL_EXTRA_FLAGS "-fsycl-unnamed-lambda;-fsycl-device-code-split=per_kernel")

add_sycl_library_unittest(syclcompat-base
  SYCL_EXTRA_FLAGS
    ${SYCL_EXTRA_FLAGS}
  SOURCES
    Device.cpp
    Dim.cpp
    Defs.cpp
  DEPENDANTS
    check-sycl-e2e-syclcompat
  )

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(usm)
set(UR_USM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_usm_test name)
    set(TEST_TARGET_NAME ${name}-test)
    add_gtest_test(${name}
        ${UR_USM_TEST_DIR}/../conformance/source/environment.cpp
        ${UR_USM_TEST_DIR}/../conformance/source/main.cpp
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::umf
        ur_testing)
    target_compile_definitions(${TEST_TARGET_NAME} PRIVATE DEVICES_ENVIRONMENT)
endfunction()

# ${PROJECT_NAME}::umf only exists when UR_UMF_REQUIRED (i.e. some enabled
# adapter actually links UMF, see source/common/CMakeLists.txt).
if(TARGET ${PROJECT_NAME}::umf)
    add_usm_test(usmPoolManager usmPoolManager.cpp)
endif()

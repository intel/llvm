# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

set(UR_CONFORMANCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_conformance_test name)
    set(TEST_TARGET_NAME test-${name})
    add_executable(${TEST_TARGET_NAME}
        ${ARGN}
        ${UR_CONFORMANCE_TEST_DIR}/source/environment.cpp
        ${UR_CONFORMANCE_TEST_DIR}/source/main.cpp)
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        GTest::gtest_main)
    add_test(NAME ${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

function(add_conformance_test_with_devices_environment name)
    add_conformance_test(${name}
        ${ARGN})
    target_compile_definitions("test-${name}" PRIVATE DEVICES_ENVIRONMENT)
endfunction()

function(add_conformance_test_with_platform_environment name)
    add_conformance_test(${name}
        ${ARGN})
    target_compile_definitions("test-${name}" PRIVATE PLATFORM_ENVIRONMENT)
endfunction()

add_subdirectory(testing)

add_subdirectory(platform)
add_subdirectory(device)
# add_subdirectory(context)
# add_subdirectory(memory)
# add_subdirectory(usm)
# add_subdirectory(event)
# add_subdirectory(queue)
# add_subdirectory(enqueue)

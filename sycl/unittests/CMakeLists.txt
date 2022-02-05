add_custom_target(SYCLUnitTests)
set_target_properties(SYCLUnitTests PROPERTIES FOLDER "SYCL tests")

foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
endforeach()

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)

include(AddSYCLUnitTest)

add_custom_target(check-sycl-unittests)

# TODO PI tests require real hardware and must be moved to intel/llvm-test-suite
option(SYCL_PI_TESTS "Enable PI-specific unit tests" ON)

if (SYCL_PI_TESTS)
  add_subdirectory(pi)
endif()

add_subdirectory(allowlist)
add_subdirectory(config)
add_subdirectory(misc)
add_subdirectory(kernel-and-program)
add_subdirectory(queue)
add_subdirectory(scheduler)
add_subdirectory(stream)
add_subdirectory(SYCL2020)
add_subdirectory(thread_safety)
add_subdirectory(program_manager)
add_subdirectory(assert)
add_subdirectory(Extensions)
add_subdirectory(windows)

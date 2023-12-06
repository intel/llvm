add_custom_target(SYCLUnitTests)
set_target_properties(SYCLUnitTests PROPERTIES FOLDER "SYCL tests")
include_directories(${BOOST_UNORDERED_INCLUDE_DIRS})

foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
endforeach()

# suppress warnings which came from Google Test sources
if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()
check_cxx_compiler_flag("-Winconsistent-missing-override" SYCL_CXX_SUPPORTS_INCONSISTENT_MISSING_OVERRIDE_FLAG)
# make CMake variable unique for SYCL toolchain by adding SYCL_ prefix to make sure there 
# are no variable overrides in the future if LLVM introduces CXX_SUPPORTS_INCONSISTENT_MISSING_OVERRIDE_FLAG
if (SYCL_CXX_SUPPORTS_INCONSISTENT_MISSING_OVERRIDE_FLAG)
  add_compile_options("-Wno-inconsistent-missing-override")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)

include(AddSYCLUnitTest)

add_custom_target(check-sycl-unittests)

# TODO PI tests require real hardware and must be moved to sycl/test-e2e.
option(SYCL_PI_TESTS "Enable PI-specific unit tests" OFF)

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
add_subdirectory(pipes)
add_subdirectory(program_manager)
add_subdirectory(assert)
add_subdirectory(Extensions)
add_subdirectory(windows)
add_subdirectory(event)
add_subdirectory(buffer)
add_subdirectory(context_device)
add_subdirectory(accessor)
add_subdirectory(handler)
add_subdirectory(builtins)
add_subdirectory(buffer/l0_specific)
# TODO Enable xpti tests for Windows
if (NOT WIN32)
  add_subdirectory(xpti_trace)
endif()

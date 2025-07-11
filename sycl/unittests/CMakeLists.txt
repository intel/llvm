add_custom_target(SYCLUnitTests)
set_target_properties(SYCLUnitTests PROPERTIES FOLDER "SYCL tests")

foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
endforeach()

add_compile_definitions(SYCL2020_DISABLE_DEPRECATION_WARNINGS SYCL_DISABLE_FSYCL_SYCLHPP_WARNING)

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

add_subdirectory(mock_opencl)

add_custom_target(check-sycl-unittests)

add_subdirectory(ur)
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
add_subdirectory(event)
add_subdirectory(buffer)
add_subdirectory(context_device)
add_subdirectory(accessor)
add_subdirectory(handler)
add_subdirectory(builtins)
add_subdirectory(buffer/l0_specific)

# Enable compression unit-tests only if zstd is present.
if (LLVM_ENABLE_ZSTD)
    add_subdirectory(compression)
endif()

# TODO Enable xpti tests for Windows
if (SYCL_ENABLE_XPTI_TRACING AND NOT WIN32)
    add_subdirectory(xpti_trace)
endif()
add_subdirectory(sampler)
add_subdirectory(reduction)
add_subdirectory(OneAPIDeviceSelector)

add_custom_target(SYCLUnitTests)
set_target_properties(SYCLUnitTests PROPERTIES FOLDER "SYCL tests")

foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
endforeach()

include(AddSYCLUnitTest)

add_subdirectory(allowlist)
add_subdirectory(misc)
add_subdirectory(pi)
add_subdirectory(kernel-and-program)
add_subdirectory(queue)
add_subdirectory(scheduler)
add_subdirectory(SYCL2020)
add_subdirectory(thread_safety)
add_subdirectory(program_manager)

add_custom_target(SYCLUnitTests)
set_target_properties(SYCLUnitTests PROPERTIES FOLDER "SYCL tests")

foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
endforeach()

# add_sycl_unittest(test_dirname SHARED|OBJECT file1.cpp, file2.cpp ...)
#
# Will compile the list of files together and link against SYCL.
# Produces a binary names `basename(test_dirname)`.
function(add_sycl_unittest test_dirname link_variant)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH 1)

  if (MSVC AND CMAKE_BUILD_TYPE MATCHES "Debug")
    set(sycl_obj_target "sycld_object")
    set(sycl_so_target "sycld")
  else()
    set(sycl_obj_target "sycl_object")
    set(sycl_so_target "sycl")
  endif()

  if ("${link_variant}" MATCHES "SHARED")
    set(SYCL_LINK_LIBS ${sycl_so_target})
    add_unittest(SYCLUnitTests ${test_dirname} ${ARGN})
  else() 
    add_unittest(SYCLUnitTests ${test_dirname} 
                $<TARGET_OBJECTS:${sycl_obj_target}> ${ARGN})
    target_compile_definitions(${test_dirname} PRIVATE __SYCL_BUILD_SYCL_DLL)

    get_target_property(SYCL_LINK_LIBS ${sycl_so_target} LINK_LIBRARIES)
  endif()
  
  target_link_libraries(${test_dirname}
    PRIVATE
      LLVMTestingSupport
      OpenCL-Headers
      ${SYCL_LINK_LIBS}
    )
  target_include_directories(${test_dirname} PRIVATE SYSTEM
      ${sycl_inc_dir}
      ${SYCL_SOURCE_DIR}/source/
      ${SYCL_SOURCE_DIR}/unittests/
    )
  # LLVM gtest uses LLVM utilities that require C++-14
  # CXX_STANDARD_REQUIRED makes CXX_STANDARD a hard requirement.
  set_target_properties(${test_dirname}
    PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
    )
endfunction()

add_subdirectory(misc)
add_subdirectory(pi)
add_subdirectory(program)
add_subdirectory(scheduler)
add_subdirectory(thread_safety)

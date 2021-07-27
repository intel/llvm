# add_sycl_unittest(test_dirname SHARED|OBJECT file1.cpp, file2.cpp ...)
#
# Will compile the list of files together and link against SYCL.
# Produces a binary names `basename(test_dirname)`.
macro(add_sycl_unittest test_dirname link_variant)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH 1)

  string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
  if (MSVC AND build_type_lower MATCHES "debug")
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
  target_include_directories(${test_dirname}
    PRIVATE SYSTEM
      ${sycl_inc_dir}
      ${SYCL_SOURCE_DIR}/source/
      ${SYCL_SOURCE_DIR}/unittests/
    )
  if (UNIX)
    # These warnings are coming from Google Test code.
    target_compile_options(${test_dirname}
      PRIVATE
        -Wno-unused-parameter
        -Wno-inconsistent-missing-override
    )
  endif()
endmacro()

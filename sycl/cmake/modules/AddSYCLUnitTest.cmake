# add_sycl_unittest(test_dirname SHARED|OBJECT file1.cpp, file2.cpp ...)
#
# Will compile the list of files together and link against SYCL.
# Produces a binary names `basename(test_dirname)`.
macro(add_sycl_unittest test_dirname link_variant)
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

macro(add_sycl_unittest_with_device test_dirname link_variant)
  set(LLVM_REQUIRES_EH 1)

  if (MSVC AND CMAKE_BUILD_TYPE MATCHES "Debug")
    set(sycl_obj_target "sycld_object")
    set(sycl_so_target "sycld")
    set(XPTI_LIB xptid)
    set(WIN_CRT /MDd)
  else()
    set(sycl_obj_target "sycl_object")
    set(sycl_so_target "sycl")
    set(XPTI_LIB xpti)
    set(WIN_CRT /MD)
  endif()

  set(COMMON_OPTS
    -DGTEST_LANG_CXX11=1
    -DGTEST_HAS_TR1_TUPLE=0
    -D__SYCL_BUILD_SYCL_DLL
    -I${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
    -I${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock/include
    -I${LLVM_BINARY_DIR}/include
    -I${LLVM_MAIN_SRC_DIR}/include
    -I${PROJECT_SOURCE_DIR}/source
    -I${PROJECT_SOURCE_DIR}/unittests
    )

  if (MSVC)
    list(APPEND COMMON_OPTS ${WIN_CRT})
    list(APPEND COMMON_OPTS /EHsc)
    list(APPEND COMMON_OPTS /link)
    list(APPEND COMMON_OPTS /LIBPATH:${LLVM_BINARY_DIR}/lib)
    list(APPEND COMMON_OPTS /subsystem:console)
    list(APPEND COMMON_OPTS /INCREMENTAL:NO)

    list(APPEND EXTRA_LIBS shlwapi)
  else()
    list(APPEND EXTRA_LIBS dl)
  endif()

  if (SYCL_ENABLE_XPTI_TRACING)
    list(APPEND EXTRA_LIBS ${XPTI_LIB})
  endif()

  if ("${link_variant}" MATCHES "OBJECT")
    add_sycl_executable(${test_dirname}
      OPTIONS -nolibsycl ${COMMON_OPTS} ${LLVM_PTHREAD_LIB} ${TERMINFO_LIB}
      SOURCES ${ARGN} $<TARGET_OBJECTS:${sycl_obj_target}>
      LIBRARIES gtest_main gtest LLVMSupport LLVMTestingSupport OpenCL ${EXTRA_LIBS}
      DEPENDANTS SYCLUnitTests)
  else()
    # TODO support shared library case.
  endif()
endmacro()

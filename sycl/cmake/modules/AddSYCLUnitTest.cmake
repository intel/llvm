# add_sycl_unittest(test_dirname SHARED|OBJECT file1.cpp, file2.cpp ...)
#
# Will compile the list of files together and link against SYCL.
# Produces a binary names `basename(test_dirname)`.
macro(add_sycl_unittest test_dirname link_variant)
  # Enable exception handling for these unit tests
  set(LLVM_REQUIRES_EH ON)
  set(LLVM_REQUIRES_RTTI ON)

  get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

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
    target_compile_definitions(${test_dirname}
                               PRIVATE __SYCL_BUILD_SYCL_DLL)

    get_target_property(SYCL_LINK_LIBS ${sycl_so_target} LINK_LIBRARIES)
  endif()

  if (SYCL_ENABLE_COVERAGE)
    target_compile_options(${test_dirname} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
    target_link_options(${test_dirname} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
  endif()

  if (SYCL_ENABLE_XPTI_TRACING)
    target_compile_definitions(${test_dirname}
      PRIVATE XPTI_ENABLE_INSTRUMENTATION XPTI_STATIC_LIBRARY)
  endif()

  # check-sycl-unittests was using an old sycl library. So, to get
  # around this problem, we add the new sycl library to the PATH and
  # LD_LIBRARY_PATH on Windows and Linux respectively.
  if (WIN32)
    add_custom_target(check-sycl-${test_dirname}
        ${CMAKE_COMMAND} -E env
        LLVM_PROFILE_FILE="${SYCL_COVERAGE_PATH}/${test_dirname}.profraw"
        SYCL_CONFIG_FILE_NAME=null.cfg
        SYCL_DEVICELIB_NO_FALLBACK=1
        SYCL_CACHE_DIR="${CMAKE_BINARY_DIR}/sycl_cache"
        "PATH=${CMAKE_BINARY_DIR}/bin;$ENV{PATH}"
        ${CMAKE_CURRENT_BINARY_DIR}/${test_dirname}
        DEPENDS
        ${test_dirname}
    )
  else()
    add_custom_target(check-sycl-${test_dirname}
        ${CMAKE_COMMAND} -E env
        LLVM_PROFILE_FILE="${SYCL_COVERAGE_PATH}/${test_dirname}.profraw"
        SYCL_CONFIG_FILE_NAME=null.cfg
        SYCL_DEVICELIB_NO_FALLBACK=1
        SYCL_CACHE_DIR="${CMAKE_BINARY_DIR}/sycl_cache"
        "LD_LIBRARY_PATH=${SYCL_BINARY_DIR}/unittests/lib:${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        ${CMAKE_CURRENT_BINARY_DIR}/${test_dirname}
        DEPENDS
        ${test_dirname}
    )
  endif()

  add_dependencies(check-sycl-unittests check-sycl-${test_dirname})

  if(WIN32)
    # Windows doesn't support LD_LIBRARY_PATH, so instead we copy the mock OpenCL binary next to the test and ensure
    # that the test itself links to OpenCL (rather than through ur_adapter_opencl.dll)
    set(mock_ocl ${CMAKE_CURRENT_BINARY_DIR}/OpenCL.dll)
    add_custom_command(TARGET ${test_dirname} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mockOpenCL> ${mock_ocl}
      DEPENDS mockOpenCL
      BYPRODUCTS ${mock_ocl}
      COMMAND_EXPAND_LISTS
      )
  endif()

  target_link_libraries(${test_dirname}
    PRIVATE
      mockOpenCL
      LLVMTestingSupport
      OpenCL-Headers
      unified-runtime::mock
      ${SYCL_LINK_LIBS}
    )

  add_dependencies(${test_dirname} ur_adapter_mock mockOpenCL)

  if(SYCL_ENABLE_EXTENSION_JIT)
    target_link_libraries(${test_dirname} PRIVATE sycl-jit)
  endif(SYCL_ENABLE_EXTENSION_JIT)

  if(WIN32)
    target_link_libraries(${test_dirname} PRIVATE UnifiedRuntimeLoader ur_win_proxy_loader)
  endif()

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
  
  target_compile_definitions(${test_dirname} PRIVATE SYCL_DISABLE_FSYCL_SYCLHPP_WARNING)
endmacro()

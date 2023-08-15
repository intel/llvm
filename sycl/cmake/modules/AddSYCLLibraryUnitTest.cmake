set(SYCL_COMPATH_UNITTEST_GCC_TOOLCHAIN "" CACHE PATH "Path to GCC installation")
# add_sycl_library_unittest(test_suite_name sycl_extra_flags
#                           file1.cpp file2.cpp ...)
#
# sycl_extra_flags: Clang extra compiler flags, e.g.
#                   "-fsycl-unnamed-lambdas;-fsycl-device-code-split"
#
# Will compile the list of files together using clang.
# Produces a single binary using all the .cpp files
# named 'test_suite_name' at ${CMAKE_CURRENT_BINARY_DIR}.
macro(add_sycl_library_unittest test_suite_name)
  cmake_parse_arguments(ARG
    ""
    ""
    "SYCL_EXTRA_FLAGS;SOURCES;DEPENDANTS"
    ${ARGN})

  set(CXX_COMPILER clang++)
  if(MSVC)
    set(CXX_COMPILER clang-cl.exe)
  endif()

  set(TRIPLES "spir64-unknown-unknown")
  if (SYCL_BUILD_PI_CUDA OR (SYCL_BUILD_PI_HIP AND "${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "NVIDIA"))
    set(TRIPLES "${TRIPLES},nvptx64-nvidia-cuda")
  endif()
  # FIXME: -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch= for amd compilation
  if ((SYCL_BUILD_PI_HIP AND "${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD"))
    set(TRIPLES "${TRIPLES},amdgcn-amd-amdhsa")
  endif()
  if (SYCL_BUILD_NATIVE_CPU)
    set(TRIPLES "${TRIPLES},native_cpu")
  endif()

  set(DEVICE_COMPILER_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CXX_COMPILER})
  set(_OUTPUT_BIN ${CMAKE_CURRENT_BINARY_DIR}/${test_suite_name}Tests)
  set(_TESTS_TARGET ${test_suite_name}Tests)
  set(_BIN_TARGET ${_TESTS_TARGET}_bin)
  set(_LLVM_TARGET_DEPENDENCIES
    "llvm_gtest_main;llvm_gtest;LLVMTestingSupport;LLVMSupport;LLVMDemangle")

  if (NOT SYCL_COMPAT_UNITTEST_GCC_TOOLCHAIN STREQUAL "")
    set(_GCC_TOOLCHAIN "--gcc-toolchain=${SYCL_COMPAT_UNITTEST_GCC_TOOLCHAIN}")
  else()
    set(_GCC_TOOLCHAIN "")
  endif()

  foreach(_lib ${_LLVM_TARGET_DEPENDENCIES})
    list(APPEND _LIBRARIES $<TARGET_LINKER_FILE:${_lib}>)
  endforeach()

  # Enable exception handling on Windows
  # Appends extra libraries not available in LIBPATH
  if(WIN32)
    set(_INTERNAL_LINKER_FLAGS /link /SUBSYSTEM:CONSOLE)
    list(APPEND _INTERNAL_EXTRA_FLAGS "/EHs")
    list(APPEND _LIBRARIES $<TARGET_LINKER_FILE:sycl>)
    list(APPEND _LIBRARIES ${LLVM_LIBRARY_OUTPUT_INTDIR}/sycl-devicelib-host.lib)
  endif()

  if(UNIX)
    foreach(_lib "pthread" "dl" "ncurses")
      list(APPEND _LIBRARIES "-l${_lib}")
    endforeach()
  endif()

  get_target_property(GTEST_INCLUDES llvm_gtest INCLUDE_DIRECTORIES)
  foreach(_dir ${GTEST_INCLUDES})
    # Avoid -I when _dir contains an empty generator expression.
    list(APPEND INCLUDE_COMPILER_STRING "$<$<BOOL:${_dir}>:-I${_dir}>")
  endforeach()

  add_custom_target(${_BIN_TARGET}
    COMMAND ${DEVICE_COMPILER_EXECUTABLE} -fsycl ${ARG_SOURCES}
      -fsycl-targets=${TRIPLES}
      ${_GCC_TOOLCHAIN}
      -o ${_OUTPUT_BIN}
      ${ARG_SYCL_EXTRA_FLAGS}
      ${_INTERNAL_EXTRA_FLAGS}
      ${INCLUDE_COMPILER_STRING}
      ${_LIBRARIES}
      ${_INTERNAL_LINKER_FLAGS}
    BYPRODUCTS ${_OUTPUT_BIN}
    DEPENDS ${ARG_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND_EXPAND_LISTS)

  add_dependencies(${_BIN_TARGET} sycl-toolchain)
  foreach(_lib ${ARG_LIBRARIES})
    add_dependencies(${_BIN_TARGET} ${_TARGET_DEPENDENCIES})
  endforeach()
  foreach(_dep ${ARG_DEPENDANTS})
    add_dependencies(${_dep} ${_BIN_TARGET})
  endforeach()

  add_executable(${_TESTS_TARGET} IMPORTED GLOBAL)
  set_target_properties(${_TESTS_TARGET} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR})

endmacro()

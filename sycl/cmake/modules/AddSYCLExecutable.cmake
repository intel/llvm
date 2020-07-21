macro(add_sycl_executable ARG_TARGET_NAME)
  cmake_parse_arguments(ARG
    ""
    "OPTIONS"
    "SOURCES"
    ${ARGN})

  set(CXX_COMPILER clang++)
  if(MSVC)
      set(CXX_COMPILER clang-cl.exe)
  endif()
  set(DEVICE_COMPILER_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CXX_COMPILER})

  add_custom_target(${ARG_TARGET_NAME}_exec ALL
    COMMAND ${DEVICE_COMPILER_EXECUTABLE} -fsycl ${ARG_OPTIONS} ${ARG_SOURCES}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(${ARG_TARGET_NAME}_exec sycl clang)

  add_executable(${ARG_TARGET_NAME} IMPORTED GLOBAL)
  set_target_properties(${ARG_TARGET_NAME} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR})
endmacro()

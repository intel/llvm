macro(add_sycl_executable ARG_TARGET_NAME)
  cmake_parse_arguments(ARG
    ""
    ""
    "OPTIONS;SOURCES;LIBRARIES;DEPENDANTS"
    ${ARGN})

  set(CXX_COMPILER clang++)
  if(MSVC)
    set(CXX_COMPILER clang-cl.exe)
    set(LIB_POSTFIX ".lib")
  else()
    set(LIB_PREFIX "-l")
  endif()
  set(DEVICE_COMPILER_EXECUTABLE ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CXX_COMPILER})

  # TODO add support for target_link_libraries(... PUBLIC ...)
  foreach(_lib ${ARG_LIBRARIES})
    list(APPEND LINKED_LIBS "${LIB_PREFIX}${_lib}${LIB_POSTFIX}")
  endforeach()

  if (LLVM_ENABLE_ASSERTIONS AND NOT SYCL_DISABLE_STL_ASSERTIONS)
    if(SYCL_USE_LIBCXX)
      set(_SYCL_EXTRA_FLAGS -D_LIBCPP_DEBUG=1)
    else()
      set(_SYCL_EXTRA_FLAGS -D_GLIBCXX_ASSERTIONS=1)
    endif()
  endif()

  add_custom_target(${ARG_TARGET_NAME}_exec
    COMMAND ${DEVICE_COMPILER_EXECUTABLE} -fsycl ${ARG_SOURCES}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}
      ${LINKED_LIBS} ${ARG_OPTIONS} ${_SYCL_EXTRA_FLAGS}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND_EXPAND_LISTS)
  add_dependencies(${ARG_TARGET_NAME}_exec sycl-toolchain)
  foreach(_lib ${ARG_LIBRARIES})
    add_dependencies(${ARG_TARGET_NAME}_exec _lib)
  endforeach()

  foreach(_dep ${ARG_DEPENDANTS})
    add_dependencies(${_dep} ${ARG_TARGET_NAME}_exec)
  endforeach()


  add_executable(${ARG_TARGET_NAME} IMPORTED GLOBAL)
  set_target_properties(${ARG_TARGET_NAME} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR})
endmacro()

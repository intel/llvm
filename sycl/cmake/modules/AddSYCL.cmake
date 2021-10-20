function (add_warning_options_linux target_name)
  target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wno-deprecated-declarations)

  check_cxx_compiler_flag(-Winstantiation-after-specialization
    HAS_INST_AFTER_SPEC)
  if (HAS_INST_AFTER_SPEC)
    target_compile_options(${target_name} PRIVATE
      -Winstantiation-after-specialization)
  endif()
endfunction()

macro(add_sycl_common_options)
  if (MSVC)
    # Force dynamic CRT on Windows
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      string(REGEX REPLACE "/MDd" "" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/MTd" "" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/MD" "" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/MT" "" ${flag_var} "${${flag_var}}")
    endforeach()
  endif()

  target_include_directories(${target_name} PRIVATE ${SYCL_INCLUDES})
  target_link_libraries(${target_name} PRIVATE ${SYCL_LINK})
  if (SYCL_COMPILE_OPTIONS)
    target_compile_options(${target_name} PRIVATE ${SYCL_COMPILE_OPTIONS})
  endif()
  if (SYCL_COMPILE_DEFINITIONS)
    target_compile_definitions(${target_name} PRIVATE ${SYCL_COMPILE_DEFINITIONS})
  endif()
  if (SYCL_USE_LIBCXX)
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
      (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
      if ((NOT (DEFINED SYCL_LIBCXX_INCLUDE_PATH)) OR (NOT (DEFINED SYCL_LIBCXX_LIBRARY_PATH)))
        message(FATAL_ERROR "When building with libc++ SYCL_LIBCXX_INCLUDE_PATHS and"
                            "SYCL_LIBCXX_LIBRARY_PATH should be set")
      endif()
      target_link_libraries(${target_name} PRIVATE "-L${SYCL_LIBCXX_LIBRARY_PATH}" -Wl,-rpath,${SYCL_LIBCXX_LIBRARY_PATH} -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc)
      target_compile_options(${target_name} PRIVATE -nostdinc++)
      target_include_directories(${target_name} PRIVATE "${SYCL_LIBCXX_INCLUDE_PATH}")
    else()
      message(FATAL_ERROR "Build with libc++ is not yet supported for this compiler")
    endif()
  else()
    # Workaround for bug in GCC version 5 and higher.
    # More information https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1568899
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
      target_link_libraries(${target_name} PRIVATE gcc_s gcc)
    endif()
  endif()

  if (UNIX)
    add_warning_options_linux(${target_name})
  endif()
endmacro()

function (add_sycl_library target_name lib_variant)
  set(SYCL_OPTIONS
    "HIDE_SYMBOLS"
  )
  set(SYCL_MULTI_VALUE
    "SOURCES"
    "LINK"
    "INCLUDES"
    "COMPILE_OPTIONS"
    "COMPILE_DEFINITIONS"
    "LINKER_SCRIPTS"
  )
  set(SYCL_SINGLE_VALUE
    "VERSION_SCRIPT"
  )

  cmake_parse_arguments(SYCL "${SYCL_OPTIONS}" "${SYCL_SINGLE_VALUE}" "${SYCL_MULTI_VALUE}" ${ARGN})

  add_library(${target_name} ${lib_variant} ${SYCL_SOURCES})
  add_sycl_common_options()

  if (UNIX)
    if (SYCL_HIDE_SYMBOLS)
      target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    endif()
    foreach(__linker_script ${SYCL_LINKER_SCRIPTS})
      set_target_properties(${target_name} PROPERTIES LINK_DEPENDS
        ${__linker_script})
      target_link_libraries(
        ${target_name} PRIVATE "-Wl,${__linker_script}")
    endforeach()
    if (SYCL_VERSION_SCRIPT)
      target_link_libraries(
        ${target_name} PRIVATE "-Wl,--version-script=${SYCL_VERSION_SCRIPT}")
      set_target_properties(${target_name} PROPERTIES LINK_DEPENDS
        ${SYCL_VERSION_SCRIPT})
    endif()
  endif()

  set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

function (add_sycl_executable target_name)
  set(SYCL_MULTI_VALUE
    "SOURCES"
    "LINK"
    "INCLUDES"
    "COMPILE_OPTIONS"
    "COMPILE_DEFINITIONS"
  )

  cmake_parse_arguments(SYCL "" "" "${SYCL_MULTI_VALUE}" ${ARGN})
  add_executable(${target_name} ${SYCL_SOURCES})
  add_sycl_common_options()
endfunction()

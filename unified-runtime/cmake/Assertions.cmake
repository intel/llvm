# From the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is lifted from llvm's LLVM_ENABLE_ASSERTIONS implementation
# https://github.com/llvm/llvm-project/blob/6be0e979896f7dd610abf263f845c532f1be3762/llvm/cmake/modules/HandleLLVMOptions.cmake#L89
if(UR_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds
  if( NOT MSVC )
    add_compile_definitions(_DEBUG)
  endif()
  # On non-Debug builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
    if (MSVC)
      # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
      foreach (flags_var_to_scrub
          CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_RELWITHDEBINFO
          CMAKE_CXX_FLAGS_MINSIZEREL
          CMAKE_C_FLAGS_RELEASE
          CMAKE_C_FLAGS_RELWITHDEBINFO
          CMAKE_C_FLAGS_MINSIZEREL)
        string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
          "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
      endforeach()
    endif()
  endif()
endif()

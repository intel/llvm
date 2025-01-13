add_llvm_library(sycl-jit-common
  lib/NDRangesHelper.cpp

  LINK_COMPONENTS
   Support
)

target_compile_options(sycl-jit-common PRIVATE ${SYCL_JIT_WARNING_FLAGS})

# Mark LLVM headers as system headers to ignore warnigns in them. This
# classification remains intact even if the same path is added as a normal
# include path in GCC and Clang.
target_include_directories(sycl-jit-common
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(sycl-jit-common
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

add_dependencies(sycl-jit-common sycl-headers)

if (BUILD_SHARED_LIBS)
  if(NOT MSVC AND NOT APPLE)
    # Manage symbol visibility through the linker to make sure no LLVM symbols
    # are exported and confuse the drivers.
    set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
    target_link_libraries(
      sycl-jit-common PRIVATE "-Wl,--version-script=${linker_script}")
    set_target_properties(sycl-jit-common
      PROPERTIES
      LINK_DEPENDS
      ${linker_script})
  endif()
endif()

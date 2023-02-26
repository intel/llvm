add_llvm_library(sycl-fusion-common
  lib/NDRangesHelper.cpp
)

target_include_directories(sycl-fusion-common
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

if (BUILD_SHARED_LIBS)
  if(NOT MSVC AND NOT APPLE)
    # Manage symbol visibility through the linker to make sure no LLVM symbols
    # are exported and confuse the drivers.
    set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
    target_link_libraries(
      sycl-fusion-common PRIVATE "-Wl,--version-script=${linker_script}")
    set_target_properties(sycl-fusion-common
      PROPERTIES
      LINK_DEPENDS
      ${linker_script})
  endif()
endif()

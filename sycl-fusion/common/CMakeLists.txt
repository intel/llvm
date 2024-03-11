add_library(sycl-fusion-common
  lib/NDRangesHelper.cpp
)

llvm_map_components_to_libnames(llvm_libs
  Support
)

target_compile_options(sycl-fusion-common PRIVATE ${SYCL_FUSION_WARNING_FLAGS})

target_include_directories(sycl-fusion-common
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(sycl-fusion-common
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(sycl-fusion-common PRIVATE ${llvm_libs})

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

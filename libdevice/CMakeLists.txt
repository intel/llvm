# Utility project providing various functionalities for SPIR-V devices
# without native support of these functionalities.

include(${CMAKE_CURRENT_SOURCE_DIR}/../libclc/cmake/modules/AddLibclc.cmake)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

if(EXISTS ${FETCHCONTENT_BASE_DIR}/unified-runtime-src)
  set(UR_SANITIZER_INCLUDE_DIR
      ${FETCHCONTENT_BASE_DIR}/unified-runtime-src/source/loader/layers/sanitizer)
elseif(EXISTS ${SYCL_UR_SOURCE_DIR})
  set(UR_SANITIZER_INCLUDE_DIR
      ${SYCL_UR_SOURCE_DIR}/source/loader/layers/sanitizer)
else()
  message(WARNING "Unified Runtime source directory is not provided, \
                   will not build libdevice sanitizer")
endif()

if(NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR})
  get_host_tool_path(clang CLANG clang_exe clang_target)
  get_host_tool_path(llvm-ar LLVM_AR llvm-ar_exe llvm-ar_target)
  get_host_tool_path(append-file APPEND_FILE append-file_exe append-file_target)
  get_host_tool_path(clang-offload-bundler CLANG_OFFLOAD_BUNDLER clang-offload-bundler_exe clang-offload-bundler_target)
  get_host_tool_path(clang-offload-packager CLANG_OFFLOAD_PACKAGER clang-offload-packager_exe clang-offload-packager_target)
  get_host_tool_path(file-table-tform FILE_TABLE_TFORM file-table-tform_exe file-table-tform_target)
  get_host_tool_path(llvm-foreach LLVM_FOREACH llvm-foreach_exe llvm-foreach_target)
  get_host_tool_path(llvm-spirv LLVM_SPIRV llvm-spirv_exe llvm-spirv_target)
  get_host_tool_path(sycl-post-link SYCL_POST_LINK sycl-post-link_exe sycl-post-link_target)
else()
  foreach(tool IN ITEMS clang llvm-ar append-file clang-offload-bundler clang-offload-packager file-table-tform llvm-foreach llvm-spirv sycl-post-link)
    find_program(LLVM_CUSTOM_TOOL_${tool} ${tool}
      PATHS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
    set(${tool}_exe ${LLVM_CUSTOM_TOOL_${tool}})
    set(${tool}_target)
  endforeach()
endif()

# Build libdevice for SYCL.
include(SYCLLibdevice)

set(LIBDEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBDEVICE_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(LIBDEVICE_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(LIBDEVICE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(test)

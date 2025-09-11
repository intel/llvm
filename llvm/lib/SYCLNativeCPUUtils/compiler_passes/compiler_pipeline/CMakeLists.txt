add_llvm_component_library(LLVMNativeCPUPipeline
  ${CMAKE_CURRENT_SOURCE_DIR}/source/attributes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/barrier_regions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/builtin_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/cl_builtin_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/define_mux_builtins_pass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/dma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/encode_kernel_metadata_pass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/group_collective_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/mangling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/metadata.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/mux_builtin_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/pass_functions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/optimal_builtin_replacement_pass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/pass_machinery.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/prepare_barriers_pass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/replace_local_module_scope_variables_pass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/scheduling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/sub_group_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/target_extension_types.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/work_item_loops_pass.cpp

  LINK_COMPONENTS
  Passes
  Core
  )

# TODO: Move to under LLVM include and work out why ADDITIONAL_HEADER_DIRS
# does not capture it.
target_include_directories(LLVMNativeCPUPipeline PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

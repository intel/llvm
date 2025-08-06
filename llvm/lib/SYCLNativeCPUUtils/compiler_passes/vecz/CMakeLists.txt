set(VECZ_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VECZ_PRIVATE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(VECZ_PRIVATE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/include)

set(COMMON_SRCS
  ${VECZ_PUBLIC_INCLUDE_DIR}/vecz/pass.h
  ${VECZ_PUBLIC_INCLUDE_DIR}/vecz/vecz_choices.h
  ${VECZ_PUBLIC_INCLUDE_DIR}/vecz/vecz_target_info.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/control_flow_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/divergence_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/instantiation_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/liveness_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/packetization_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/simd_width_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/stride_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/uniform_value_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/vectorizable_function_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/analysis/vectorization_unit_analysis.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/common_gep_elimination_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/control_flow_conversion_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/inline_post_vectorization_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/instantiation_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/interleaved_group_combine_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/packetization_helpers.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/packetization_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/packetizer.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/passes.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/printf_scalarizer.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/scalarization_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/scalarizer.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/transform/ternary_transform_pass.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/control_flow_boscc.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/control_flow_roscc.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/debugging.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/ir_cleanup.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/llvm_helpers.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/memory_operations.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/offset_info.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/reachability.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/simd_packet.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/vectorization_context.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/vectorization_helpers.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/vectorization_heuristics.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/vectorization_unit.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/vectorizer.h
  ${VECZ_PRIVATE_INCLUDE_DIR}/vecz_pass_builder.h
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/control_flow_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/divergence_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/instantiation_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/liveness_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/packetization_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/simd_width_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/stride_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/uniform_value_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/vectorizable_function_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/analysis/vectorization_unit_analysis.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/basic_mem2reg_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/builtin_inlining_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/common_gep_elimination_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/control_flow_conversion_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/inline_post_vectorization_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/loop_rotate_custom_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/instantiation_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/interleaved_group_combine_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/packetization_helpers.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/packetization_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/packetizer.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/passes.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/pre_linearize_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/printf_scalarizer.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/remove_intptr_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/scalarization_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/scalarizer.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/simplify_infinite_loop_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/squash_small_vectors_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/ternary_transform_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/transform/uniform_reassociation_pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/control_flow_boscc.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/control_flow_roscc.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/debugging.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/ir_cleanup.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/llvm_helpers.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/memory_operations.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/offset_info.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/pass.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/reachability.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/simd_packet.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vector_target_info.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vector_target_info_arm.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vector_target_info_riscv.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vectorization_choices.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vectorization_context.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vectorization_helpers.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vectorization_heuristics.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vectorization_unit.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vectorizer.cpp
  ${VECZ_PRIVATE_SOURCE_DIR}/vecz_pass_builder.cpp
)

if(MSVC)
  # Disable: unreferenced formal parameter.
  list(REMOVE_ITEM VECZ_COMPILE_OPTIONS -we4100)
  list(APPEND VECZ_COMPILE_OPTIONS -wd4100)
endif()

add_llvm_component_library(LLVMNativeCPUVecz
  ${COMMON_SRCS}
  LINK_COMPONENTS
  NativeCPUPipeline
  support
  core
  analysis
  instcombine
  aggressiveinstcombine
  transformutils
  scalaropts
  ipo
  passes
  )

target_include_directories(LLVMNativeCPUVecz
  PUBLIC $<BUILD_INTERFACE:${VECZ_PUBLIC_INCLUDE_DIR}>
  PRIVATE $<BUILD_INTERFACE:${VECZ_PRIVATE_INCLUDE_DIR}>
)
target_compile_options(LLVMNativeCPUVecz PRIVATE ${VECZ_COMPILE_OPTIONS})
target_compile_definitions(LLVMNativeCPUVecz PRIVATE
  ${VECZ_COMPILE_DEFINITIONS})

# Currently disabled by default, these allow us to run lit tests using veczc
set(BUILD_VECZ_TEST_TOOLS OFF CACHE BOOL "Build vecz test and tools" FORCE)
if (BUILD_VECZ_TEST_TOOLS)
  add_subdirectory(tools)
  add_subdirectory(test)
endif()

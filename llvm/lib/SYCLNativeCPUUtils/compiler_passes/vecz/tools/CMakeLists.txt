llvm_map_components_to_libnames(llvm_libs all ${LLVM_TARGETS_TO_BUILD})
list(REMOVE_ITEM llvm_libs LTO OptRemarks)

add_llvm_tool(veczc
 ${CMAKE_CURRENT_SOURCE_DIR}/source/veczc.cpp
)
target_compile_options(veczc PRIVATE ${VECZ_COMPILE_OPTIONS})
target_compile_definitions(veczc PRIVATE ${VECZ_COMPILE_DEFINITIONS})
target_include_directories(veczc PRIVATE
 ${CMAKE_CURRENT_SOURCE_DIR}/include>
 ${CMAKE_CURRENT_SOURCE_DIR}/../../compiler_pipeline/include
 ${CMAKE_CURRENT_SOURCE_DIR}/../../vecz/include
 )
target_link_libraries(veczc PUBLIC ${llvm_libs})

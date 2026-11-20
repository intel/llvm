add_library(offload_sanitizer_msan OBJECT
  sanitizer_options.cpp
  sanitizer_stackdepot.cpp
  sanitizer_stacktrace.cpp
  msan_report.cpp
)
llvm_update_compile_flags(offload_sanitizer_msan)
target_compile_options(offload_sanitizer_msan PRIVATE ${offload_compile_flags})
target_compile_definitions(offload_sanitizer_msan PRIVATE
  TARGET_NAME=MSAN
  DEBUG_PREFIX="MSAN"
)
target_include_directories(offload_sanitizer_msan PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
)
if(LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(offload_sanitizer_msan INTERFACE LLVM)
else()
  llvm_map_components_to_libnames(msan_llvm_libs Symbolize Support)
  target_link_libraries(offload_sanitizer_msan INTERFACE ${msan_llvm_libs})
endif()
set_target_properties(offload_sanitizer_msan PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

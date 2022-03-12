add_executable(sycl-trace main.cpp)

target_include_directories(sycl-trace PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
)

target_compile_options(sycl-trace PRIVATE -fno-exceptions -fno-rtti)

link_llvm_libs(sycl-trace
  LLVMSupport
)

add_library(sycl_pi_trace_collector SHARED
  collector.cpp
  pi_trace_collector.cpp
  ze_trace_collector.cpp
)

# To get L0 loader
add_dependencies(sycl_pi_trace_collector pi_level_zero)

if (NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
  set(LEVEL_ZERO_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/../../plugins/level_zero/level_zero_loader/include")
endif()
target_include_directories(sycl_pi_trace_collector PRIVATE
  ${LEVEL_ZERO_INCLUDE_DIR}
)

target_compile_definitions(sycl_pi_trace_collector PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(sycl_pi_trace_collector PRIVATE xptifw)
if (TARGET OpenCL-Headers)
  target_link_libraries(sycl_pi_trace_collector PRIVATE OpenCL-Headers)
endif()

target_include_directories(sycl_pi_trace_collector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
    "${sycl_inc_dir}"
    "${sycl_src_dir}"
)

add_dependencies(sycl-trace sycl_pi_trace_collector)
add_dependencies(sycl-toolchain sycl-trace)

include(GNUInstallDirs)
install(TARGETS sycl-trace sycl_pi_trace_collector
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-trace
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
)


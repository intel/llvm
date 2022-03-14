add_executable(sycl-prof
  main.cpp
)

target_include_directories(sycl-prof PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
)

link_llvm_libs(sycl-prof
  LLVMSupport
)

target_compile_options(sycl-prof PRIVATE -fno-exceptions -fno-rtti)

add_library(sycl_profiler_collector SHARED collector.cpp)
target_compile_definitions(sycl_profiler_collector PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(sycl_profiler_collector PRIVATE xptifw)
if (TARGET OpenCL-Headers)
  target_link_libraries(sycl_profiler_collector PRIVATE OpenCL-Headers)
endif()
target_include_directories(sycl_profiler_collector PRIVATE
    "${sycl_inc_dir}"
    "${sycl_src_dir}"
)

target_compile_options(sycl_profiler_collector PRIVATE -g)

add_dependencies(sycl-prof sycl_profiler_collector)
add_dependencies(sycl-toolchain sycl-prof)

include(GNUInstallDirs)
install(TARGETS sycl-prof sycl_profiler_collector
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-prof
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-prof
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-prof
)

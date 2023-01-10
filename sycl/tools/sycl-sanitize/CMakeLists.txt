add_executable(sycl-sanitize main.cpp)

target_include_directories(sycl-sanitize PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
)

link_llvm_libs(sycl-sanitize
  LLVMSupport
)

target_compile_options(sycl-sanitize PRIVATE -fno-exceptions -fno-rtti)

add_library(sycl_sanitizer_collector SHARED collector.cpp)
target_compile_definitions(sycl_sanitizer_collector PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(sycl_sanitizer_collector PRIVATE xptifw)
if (TARGET OpenCL-Headers)
  target_link_libraries(sycl_sanitizer_collector PRIVATE OpenCL-Headers)
endif()

target_include_directories(sycl_sanitizer_collector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
    "${sycl_inc_dir}"
    "${sycl_src_dir}"
)

add_dependencies(sycl-sanitize sycl_sanitizer_collector)
add_dependencies(sycl-toolchain sycl-sanitize)

include(GNUInstallDirs)
install(TARGETS sycl-sanitize sycl_sanitizer_collector
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-sanitize
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-sanitize
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-sanitize
)

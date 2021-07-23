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

add_executable(sycl-sanitizer tool.cpp)

add_dependencies(sycl-sanitizer sycl_sanitizer_collector)
add_dependencies(sycl-toolchain sycl-sanitizer)

include(GNUInstallDirs)
install(TARGETS sycl-sanitizer sycl_sanitizer_collector
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-sanitizer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-sanitizer
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-sanitizer
)

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

find_package(Python3 REQUIRED)

add_custom_target(pi-pretty-printers
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_pi_pretty_printers.py
  ${sycl_inc_dir}/CL/sycl/detail/pi.h
  SOURCES ${sycl_inc_dir}/CL/sycl/detail/pi.h
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/pi_structs.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/pi_printers.def
  )

# To get L0 loader
add_dependencies(sycl_pi_trace_collector pi_level_zero)

target_link_libraries(sycl_pi_trace_collector PRIVATE LevelZeroLoader::Headers)

add_custom_target(ze-pretty-printers
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_ze_pretty_printers.py
  ${SYCL_INCLUDE_BUILD_DIR}/sycl/level_zero/ze_api.h
  DEPENDS pi_level_zero
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/ze_printers.def
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
    "${CMAKE_CURRENT_BINARY_DIR}"
)

add_dependencies(sycl_pi_trace_collector pi-pretty-printers ze-pretty-printers)
add_dependencies(sycl-trace sycl_pi_trace_collector)
add_dependencies(sycl-toolchain sycl-trace)

include(GNUInstallDirs)
install(TARGETS sycl-trace sycl_pi_trace_collector
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-trace
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
)


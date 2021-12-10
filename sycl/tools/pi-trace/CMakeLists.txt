add_library(pi_trace SHARED pi_trace.cpp)
target_compile_definitions(pi_trace PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(pi_trace PRIVATE xptifw)
if (TARGET OpenCL-Headers)
  target_link_libraries(pi_trace PRIVATE OpenCL-Headers)
endif()

target_include_directories(pi_trace PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
    "${sycl_inc_dir}"
    "${sycl_src_dir}"
)

add_dependencies(sycl-toolchain pi_trace)

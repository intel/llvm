add_library(pi_trace SHARED pi_trace.cpp)
target_link_libraries(pi_trace PRIVATE xptifw)
target_include_directories(pi_trace PRIVATE "${XPTI_SOURCE_DIR}/include")
target_include_directories(pi_trace PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/")
target_include_directories(pi_trace PRIVATE "${sycl_inc_dir}")
target_include_directories(pi_trace PRIVATE "${sycl_src_dir}")

target_compile_options(pi_trace PRIVATE -g -O0)

if(UNIX)
  target_link_libraries(pi_trace PRIVATE dl)
endif()

if (XPTI_ENABLE_TBB)
  target_link_libraries(pi_trace PRIVATE tbb)
endif()

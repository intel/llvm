add_library(sycl_perf_collector SHARED
  sycl_perf_collector.cpp
)
target_link_libraries(sycl_perf_collector PRIVATE
  xptifw
  ${CMAKE_DL_LIBS}
)
target_include_directories(sycl_perf_collector PRIVATE
  ${XPTIFW_DIR}/include
  ${XPTI_DIR}/include
  ${XPTIFW_DIR}/samples/include
)

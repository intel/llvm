add_library(sycl_perf_collector SHARED
  sycl_perf_collector.cpp
)
target_compile_definitions(sycl_perf_collector PUBLIC XPTI_CALLBACK_API_EXPORTS=1)
target_link_libraries(sycl_perf_collector PRIVATE
  ${xptifw_lib}
  ${CMAKE_DL_LIBS}
)
target_include_directories(sycl_perf_collector PRIVATE
  ${XPTIFW_DIR}/include
  ${XPTI_DIR}/include
  ${SAMPLES_DIR}/include
)


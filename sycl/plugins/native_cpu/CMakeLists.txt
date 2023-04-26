add_sycl_plugin(native_cpu
  SOURCES
    pi_native_cpu.cpp
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR} 
  LIBRARIES
    sycl
)

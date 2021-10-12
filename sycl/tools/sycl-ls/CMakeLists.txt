add_executable(sycl-ls sycl-ls.cpp)
add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE "${sycl_inc_dir}")

set(sycl_lib sycl)
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  set(sycl_lib sycld)
endif()

target_link_libraries(sycl-ls
  PRIVATE
    ${sycl_lib}
    OpenCL-Headers
)
install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)

set(sycl_lib sycl)
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  set(sycl_lib sycld)
endif()

add_sycl_executable(sycl-ls
  SOURCES
    sycl-ls.cpp
  INCLUDES
    "${sycl_inc_dir}"
  LINK
    ${sycl_lib}
    OpenCL-Headers
)
add_dependencies(sycl-ls sycl)

install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)
add_llvm_install_targets(install-sycl-ls
  DEPENDS sycl-ls
  COMPONENT level-zero-headers
)

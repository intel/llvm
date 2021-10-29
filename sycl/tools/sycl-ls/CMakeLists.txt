add_sycl_executable(sycl-ls
  SOURCES
    sycl-ls.cpp
  INCLUDES
    "${sycl_inc_dir}"
  LINK
    sycl-runtime
    OpenCL-Headers
)

install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)
add_llvm_install_targets(install-sycl-ls
  DEPENDS sycl-ls
  COMPONENT level-zero-headers
)

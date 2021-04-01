add_executable(sycl-ls sycl-ls.cpp)
add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE "${sycl_inc_dir}"
#FIXME: OpenCL headers has to be added explicitly here for CUDA target build,
#       to overrule OpenCL headers included in CUDA which are implicitly present
#       as sycl target interface (for CUDA build)
#       This shouldn't be required if sycl-ls app would depend solely on SYCL
#       headers, but currently it requires OpenCL headers
      "${OPENCL_ICD_LOADER_HEADERS_DIR}")
target_link_libraries(sycl-ls
  PRIVATE
    sycl
    OpenCL-Headers
)
install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)

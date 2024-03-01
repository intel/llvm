add_executable(sycl-ls sycl-ls.cpp)
add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE "${sycl_inc_dir}")

set(sycl_lib sycl)
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  set(sycl_lib sycld)
endif()

# Disable aspect::image warning.
target_compile_definitions(sycl-ls PRIVATE SYCL_DISABLE_IMAGE_ASPECT_WARNING)

target_link_libraries(sycl-ls
  PRIVATE
    ${sycl_lib}
    OpenCL-Headers
)
if (WIN32)
  # 0x900: Search for the dependency DLLs only in the System32 directory and in the directory with sycl-ls.exe
  target_link_options(sycl-ls PRIVATE /DEPENDENTLOADFLAG:0x900)
endif()
install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)

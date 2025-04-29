add_executable(syclbin-dump syclbin-dump.cpp)

link_llvm_libs(syclbin-dump LLVMSupport LLVMObject)
target_include_directories(syclbin-dump PRIVATE
    "${sycl_inc_dir}"
)

set(sycl_lib sycl)
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  set(sycl_lib sycld)
endif()
target_link_libraries(syclbin-dump PRIVATE ${sycl_lib})

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  target_compile_options(syclbin-dump PRIVATE -fno-rtti)
endif()

add_dependencies(sycl-toolchain syclbin-dump)

install(TARGETS syclbin-dump
  RUNTIME DESTINATION "bin" COMPONENT syclbin-dump)

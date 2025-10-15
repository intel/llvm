set(sycl_lib_dir $<TARGET_FILE_DIR:sycl>)
add_definitions(-DSYCL_LIB_DIR="${sycl_lib_dir}")
# https://github.com/intel/llvm/issues/19626
if(NOT LLVM_LIBCXX_USED)
add_sycl_unittest(MiscTests SHARED
  CircularBuffer.cpp
  OsUtils.cpp
  PropertyUtils.cpp
  DetailString.cpp
)
endif()
add_subdirectory(LinkGraph)

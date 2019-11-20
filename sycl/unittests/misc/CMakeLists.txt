set(CMAKE_CXX_EXTENSIONS OFF)

set(sycl_lib_dir $<TARGET_FILE_DIR:sycl>)
add_definitions(-DSYCL_LIB_DIR="${sycl_lib_dir}")
add_sycl_unittest(MiscTests
  OsUtils.cpp
)

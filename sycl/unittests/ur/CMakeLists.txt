set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest_both(UrTests UrTestsPreview OBJECT
  UrUtility.cpp
)

add_dependencies(UrTests sycl)
target_include_directories(UrTests PRIVATE SYSTEM ${sycl_inc_dir})
add_dependencies(UrTestsPreview sycl)
target_include_directories(UrTestsPreview PRIVATE SYSTEM ${sycl_inc_dir})

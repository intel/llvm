set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(UrTests OBJECT
  UrUtility.cpp
)

add_dependencies(UrTests_non_preview sycl)
target_include_directories(UrTests_non_preview PRIVATE SYSTEM ${sycl_inc_dir})
add_dependencies(UrTests_preview sycl)
target_include_directories(UrTests_preview PRIVATE SYSTEM ${sycl_inc_dir})

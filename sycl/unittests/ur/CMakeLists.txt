set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(UrTests OBJECT
  UrUtility.cpp
)

add_dependencies(UrTests sycl)
target_include_directories(UrTests PRIVATE SYSTEM ${sycl_inc_dir})

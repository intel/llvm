set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(UrTests OBJECT
  UrUtility.cpp
)

add_dependencies(UrTests_Non_Preview_Tests sycl)
target_include_directories(UrTests_Non_Preview_Tests PRIVATE SYSTEM ${sycl_inc_dir})
add_dependencies(UrTests_Preview_Tests sycl)
target_include_directories(UrTests_Preview_Tests PRIVATE SYSTEM ${sycl_inc_dir})

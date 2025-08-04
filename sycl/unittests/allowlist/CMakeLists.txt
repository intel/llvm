set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest_both(AllowListTests AllowListTestsPreview OBJECT
    ParseAllowList.cpp
    DeviceIsAllowed.cpp
)

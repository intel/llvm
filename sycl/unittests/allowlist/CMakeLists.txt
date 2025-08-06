set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittests(AllowListTests OBJECT
    ParseAllowList.cpp
    DeviceIsAllowed.cpp
)

set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(AllowListTests OBJECT 
    ParseAllowList.cpp
    DeviceIsAllowed.cpp
)

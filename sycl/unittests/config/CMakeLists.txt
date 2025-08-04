set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest_both(ConfigTests ConfigTestsPreview OBJECT
    ConfigTests.cpp
    PreferredWGSizeConfigTests.cpp
)

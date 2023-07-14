
add_conformance_test_with_devices_environment(adapter-cuda 
    cuda_fixtures.h
    cuda_urContextGetNativeHandle.cpp
    cuda_urDeviceGetNativeHandle.cpp
    cuda_urDeviceCreateWithNativeHandle.cpp
    cuda_urEventGetNativeHandle.cpp
    cuda_urEventCreateWithNativeHandle.cpp
)
target_link_libraries(test-adapter-cuda PRIVATE cudadrv)

set_tests_properties(adapter-cuda PROPERTIES
        LABELS "conformance:cuda"
        ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_cuda>\""
    )

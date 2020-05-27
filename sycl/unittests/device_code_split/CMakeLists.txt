add_device_enabled_unittest(TESTNAME DeviceCodeSplitPerSource
    HOST_SOURCES "src/split-per-source.cpp"
    TARGET_FLAGS -fsycl-device-code-split=per_source -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice
    DEVICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/device-code-first-file.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/device-code-second-file.cpp"
)

add_device_enabled_unittest(TESTNAME DeviceCodeSplitPerKernel
    HOST_SOURCES "src/split-per-kernel.cpp"
    TARGET_FLAGS -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice
    DEVICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/device-code-first-file.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/device-code-second-file.cpp"
)

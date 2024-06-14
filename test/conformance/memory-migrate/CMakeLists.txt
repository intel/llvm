if (UR_DPCXX)
    add_conformance_test_with_kernels_environment(memory-migrate
        urMemBufferMigrateAcrossDevices.cpp)
endif()

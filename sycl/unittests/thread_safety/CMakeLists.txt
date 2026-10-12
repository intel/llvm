# https://github.com/intel/llvm/issues/19591
if (NOT LLVM_LIBCXX_USED)
add_sycl_unittest(ThreadSafetyTests OBJECT
    DeviceGlobalProgramHandle.cpp
    HostAccessorDeadLock.cpp
    InteropKernelEnqueue.cpp
)
endif()

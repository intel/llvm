# https://github.com/intel/llvm/issues/19591
if (NOT LLVM_LIBCXX_USED)
add_sycl_unittest_both(ThreadSafetyTests ThreadSafetyTestsPreview OBJECT
    HostAccessorDeadLock.cpp
    InteropKernelEnqueue.cpp
)
endif()

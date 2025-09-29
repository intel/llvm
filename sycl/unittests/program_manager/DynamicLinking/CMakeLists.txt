# These tests introduce images with unresolved dependencies, which can affect other tests if they use kernel bundle API.
# TODO this can be merged back into program manager tests once __sycl_unregister_lib is implemented.
# https://github.com/intel/llvm/issues/19597
if (NOT LLVM_LIBCXX_USED)
add_sycl_unittest(DynamicLinkingTests OBJECT
  DynamicLinking.cpp
)
endif()

add_sycl_unittest(IPCTests OBJECT
  Memory.cpp
)

set_target_properties(IPCTests_preview PROPERTIES CXX_STANDARD 20)
set_target_properties(IPCTests_non_preview PROPERTIES CXX_STANDARD 20)

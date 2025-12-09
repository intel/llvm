add_sycl_unittest(IPCTests OBJECT
  Memory.cpp
)

set_target_properties(IPCTests_Preview_Tests PROPERTIES CXX_STANDARD 20)
set_target_properties(IPCTests_Non_Preview_Tests PROPERTIES CXX_STANDARD 20)

if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  add_sycl_unittest(IPCTests OBJECT
    Memory.cpp
  )

  target_compile_features(IPCTests_Preview_Tests PUBLIC cxx_std_20)
  target_compile_features(IPCTests_Non_Preview_Tests PUBLIC cxx_std_20)
endif()

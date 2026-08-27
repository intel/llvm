if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  add_sycl_unittest(IPCTests OBJECT
    Event.cpp
    Memory.cpp
    PhysicalMemory.cpp
  )

  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    target_compile_features(IPCTests-Preview_Tests PUBLIC cxx_std_20)
    # Pre-existing [=] implicit-this capture in device_impl.hpp.
    target_compile_options(IPCTests-Preview_Tests PRIVATE
      $<$<CXX_COMPILER_ID:Clang,GNU>:-Wno-deprecated-this-capture>)
  endif()
  target_compile_features(IPCTests-Non_Preview_Tests PUBLIC cxx_std_20)
  # Pre-existing [=] implicit-this capture in device_impl.hpp.
  target_compile_options(IPCTests-Non_Preview_Tests PRIVATE
    $<$<CXX_COMPILER_ID:Clang,GNU>:-Wno-deprecated-this-capture>)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

# https://github.com/intel/llvm/issues/19597
if (NOT LLVM_LIBCXX_USED)
add_sycl_unittests(ProgramManagerTests OBJECT
  CompileTarget.cpp
  BuildLog.cpp
  itt_annotations.cpp
  SubDevices.cpp
  passing_link_and_compile_options.cpp
  Cleanup.cpp
  MultipleDevsKernelBundle.cpp
)
endif()

add_subdirectory(arg_mask)
add_subdirectory(DynamicLinking)

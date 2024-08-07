set(CMAKE_CXX_EXTENSIONS OFF)
add_sycl_unittest(ProgramManagerTests OBJECT
  BuildLog.cpp
  DynamicLinking.cpp
  itt_annotations.cpp
  SubDevices.cpp
  passing_link_and_compile_options.cpp
)

add_subdirectory(arg_mask)


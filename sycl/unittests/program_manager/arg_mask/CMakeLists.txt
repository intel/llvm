set(CMAKE_CXX_EXTENSIONS OFF)
add_sycl_unittest_both(ProgramManagerArgMaskTests ProgramManagerArgMaskTestsPreview OBJECT
  EliminatedArgMask.cpp
)


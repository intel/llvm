set(CMAKE_CXX_EXTENSIONS OFF)
add_sycl_unittest(ProgramManagerTests OBJECT
  PARTIAL_SOURCES_INTENDED # FIXME: Remove once below is fixed.
  BuildLog.cpp
  # FIXME: Depends on https://github.com/oneapi-src/unified-runtime/pull/1861
  # DynamicLinking.cpp
  itt_annotations.cpp
  SubDevices.cpp
  passing_link_and_compile_options.cpp
)

add_subdirectory(arg_mask)


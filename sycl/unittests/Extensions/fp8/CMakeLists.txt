add_sycl_unittest(FP8TypesTests OBJECT
  fp8_e4m3.cpp
  fp8_e5m2.cpp
  fp8_e8m0.cpp
)

target_compile_options(FP8TypesTests_Preview_Tests PUBLIC -D__SYCL_TARGET_INTEL_GPU_CRI__)
target_compile_options(FP8TypesTests_Non_Preview_Tests PUBLIC -D__SYCL_TARGET_INTEL_GPU_CRI__)


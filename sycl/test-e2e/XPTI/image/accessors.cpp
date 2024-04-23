// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %{build} -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %{run} %t.out | FileCheck %s

#ifdef XPTI_COLLECTOR

#include "../Inputs/memory_info_collector.cpp"

#else

#include <sycl/detail/core.hpp>

using namespace sycl::access;

int main() {
  sycl::queue Q{};

  /* Unsampled images */
  {
    // CHECK:{{[0-9]+}}|Create unsampled image|[[UIMGID:[0-9,a-f,x]+]]|0x{{[0-9,a-f]+}}|1|{3,0,0}|7|{{.*}}accessors.cpp:[[# @LINE + 1]]:30
    sycl::unsampled_image<1> UImg(sycl::image_format::r32b32g32a32_uint, 3);

    Q.submit([&](sycl::handler &CGH) {
      // CHECK:{{[0-9]+}}|Construct unsampled image accessor|[[UIMGID]]|0x{{[0-9,a-f]+}}|1|1024|{{.*}}vec{{.*}}|16|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
      auto A1 = UImg.get_access<sycl::uint4, mode::read,
                                sycl::image_target::host_task>(CGH);
    });

    Q.submit([&](sycl::handler &CGH) {
      // CHECK:{{[0-9]+}}|Construct unsampled image accessor|[[UIMGID]]|0x{{[0-9,a-f]+}}|1|1025|{{.*}}vec{{.*}}|16|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
      auto A1 = UImg.get_access<sycl::uint4, mode::write,
                                sycl::image_target::host_task>(CGH);
    });

    // CHECK:{{[0-9]+}}|Construct host unsampled image accessor|[[UIMGID]]|0x{{[0-9,a-f]+}}|1024|{{.*}}vec{{.*}}|16|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
    { auto HA = UImg.get_host_access<sycl::int4, mode::read>(); }
    // CHECK:{{[0-9]+}}|Construct host unsampled image accessor|[[UIMGID]]|0x{{[0-9,a-f]+}}|1025|{{.*}}vec{{.*}}|16|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
    { auto HA = UImg.get_host_access<sycl::float4, mode::write>(); }
    // CHECK:{{[0-9]+}}|Construct host unsampled image accessor|[[UIMGID]]|0x{{[0-9,a-f]+}}|1026|{{.*}}vec{{.*}}|8|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
    { auto HA = UImg.get_host_access<sycl::half4, mode::read_write>(); }
  }
  // CHECK:{{[0-9]+}}|Destruct image|[[UIMGID]]

  /* Sampled images */
  {
    constexpr size_t SampledImgElemCount = 3;
    int16_t SampledImgData[4 * SampledImgElemCount] = {0};
    sycl::image_sampler Sampler{sycl::addressing_mode::repeat,
                                sycl::coordinate_normalization_mode::normalized,
                                sycl::filtering_mode::linear};

    // CHECK:{{[0-9]+}}|Create sampled image|[[SIMGID:[0-9,a-f,x]+]]|0x{{[0-9,a-f]+}}|1|{3,0,0}|3|4403|1|4417|{{.*}}accessors.cpp:[[# @LINE + 1]]:28
    sycl::sampled_image<1> SImg(SampledImgData,
                                sycl::image_format::r16g16b16a16_sint, Sampler,
                                SampledImgElemCount);

    Q.submit([&](sycl::handler &CGH) {
      // CHECK:{{[0-9]+}}|Construct sampled image accessor|[[SIMGID]]|0x{{[0-9,a-f]+}}|1|{{.*}}vec{{.*}}|16|{{.*}}accessors.cpp:[[# @LINE + 2]]:11
      auto A1 =
          SImg.get_access<sycl::uint4, sycl::image_target::host_task>(CGH);
    });

    // CHECK:{{[0-9]+}}|Construct host sampled image accessor|[[SIMGID]]|0x{{[0-9,a-f]+}}|{{.*}}vec{{.*}}|16|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
    { auto HA = SImg.get_host_access<sycl::int4>(); }
    // CHECK:{{[0-9]+}}|Construct host sampled image accessor|[[SIMGID]]|0x{{[0-9,a-f]+}}|{{.*}}vec{{.*}}|8|{{.*}}accessors.cpp:[[# @LINE + 1]]:17
    { auto HA = SImg.get_host_access<sycl::half4>(); }
  }
  // CHECK:{{[0-9]+}}|Destruct image|[[SIMGID]]

  return 0;
}
#endif

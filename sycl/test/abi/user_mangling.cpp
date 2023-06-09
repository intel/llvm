// clang-format off
// RUN: %clangxx -fsycl -c -emit-llvm -D__SYCL_INTERNAL_API -S -o - %s | FileCheck %s --check-prefix CHK-HOST
// RUN: %clangxx -fsycl -fsycl-device-only -D__SYCL_INTERNAL_API -O0 -c -emit-llvm -S -o - %s | FileCheck %s --check-prefix CHK-DEVICE
// REQUIRES: linux
// UNSUPPORTED: libcxx

#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
// CHK-DEVICE: define dso_local spir_func void @_Z4accdN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void accd(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::device>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::local>) {} 

// CHK_DEVICE: define dso_local void @_Z3accN4sycl3_V114local_accessorIiLi1EEE({{.*}})
SYCL_EXTERNAL void acc(sycl::local_accessor<int, 1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z3accN4sycl3_V18accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2017ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void acc(sycl::accessor<sycl::vec<sycl::opencl::cl_int, 4>, 1, sycl::access::mode::read, sycl::access::target::image>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z11private_memN4sycl3_V114private_memoryIiLi1EEE({{.*}})
SYCL_EXTERNAL void private_mem(sycl::private_memory<int, 1>) {};

// CHK-DEVICE: define dso_local spir_func void @_Z5rangeN4sycl3_V15rangeILi1EEE({{.*}})
SYCL_EXTERNAL void range(sycl::range<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z2idN4sycl3_V12idILi1EEE({{.*}})
SYCL_EXTERNAL void id(sycl::id<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z4itemN4sycl3_V12idILi1EEE({{.*}})
SYCL_EXTERNAL void item(sycl::id<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z3vecN4sycl3_V13vecIiLi16EEE({{.*}})
SYCL_EXTERNAL void vec(sycl::vec<int, 16>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z6streamN4sycl3_V16streamE({{.*}})
SYCL_EXTERNAL void stream(sycl::stream) {}

// CHK-DEVICE: define dso_local spir_func void @_Z7samplerN4sycl3_V17samplerE({{.*}})
SYCL_EXTERNAL void sampler(sycl::sampler) {}
#else
// CHK-HOST: define dso_local void @_Z3accN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2018ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer>) {} 

// CHK-HOST: define dso_local void @_Z3accN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::local>) {} 

// CHK-HOST: define dso_local void @_Z3accN4sycl3_V114local_accessorIiLi1EEE({{.*}})
void acc(sycl::local_accessor<int, 1>) {}

// CHK-HOST: define dso_local void @_Z3accN4sycl3_V18accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2019ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
void acc(sycl::accessor<sycl::vec<sycl::opencl::cl_int, 4>, 1, sycl::access::mode::read, sycl::access::target::host_image>) {} 

// CHK-HOST: define dso_local void @_Z3bufN4sycl3_V16bufferIiLi1ENS0_6detail17aligned_allocatorIiEEvEE({{.*}})
void buf(sycl::buffer<int>) {}

// CHK-HOST: define dso_local void @_Z3ctxN4sycl3_V17contextE({{.*}})
void ctx(sycl::context) {}

// CHK-HOST: define dso_local void @_Z6deviceN4sycl3_V16deviceE({{.*}})
void device(sycl::device) {}

// CHK-HOST: define dso_local void @_Z10device_evtN4sycl3_V112device_eventE({{.*}})
void device_evt(sycl::device_event) {}

// CHK-HOST: define dso_local void @_Z5eventN4sycl3_V15eventE({{.*}})
void event(sycl::event) {}

// CHK-HOST: define dso_local void @_Z15device_selectorRN4sycl3_V115device_selectorE({{.*}})
void device_selector(sycl::device_selector&) {}

// CHK-HOST: define dso_local void @_Z7handlerRN4sycl3_V17handlerE({{.*}})
void handler(sycl::handler&) {}

// CHK-HOST: define dso_local void @_Z5imageN4sycl3_V15imageILi1ENS0_6detail17aligned_allocatorIhEEEE({{.*}})
void image(sycl::image<1>) {}

// CHK-HOST: define dso_local void @_Z5rangeN4sycl3_V15rangeILi1EEE({{.*}})
void range(sycl::range<1>) {}

// CHK-HOST: define dso_local void @_Z2idN4sycl3_V12idILi1EEE({{.*}})
void id(sycl::id<1>) {}

// CHK-HOST: define dso_local void @_Z4itemN4sycl3_V14itemILi1ELb1EEE({{.*}})
void item(sycl::item<1>) {}

// CHK-HOST: define dso_local void @_Z6streamN4sycl3_V16streamE({{.*}})
void stream(sycl::stream) {}

// CHK-HOST: define dso_local void @_Z7samplerN4sycl3_V17samplerE({{.*}})
void sampler(sycl::sampler) {}

// CHK-HOST: define dso_local void @_Z5queueN4sycl3_V15queueE({{.*}})
void queue(sycl::queue) {}

// CHK-HOST: define dso_local void @_Z6kernelN4sycl3_V16kernelE({{.*}})
void kernel(sycl::kernel) {}

// CHK-HOST: define dso_local void @_Z8platformN4sycl3_V18platformE({{.*}})
void platform(sycl::platform) {}

// CHK-HOST: define dso_local void @_Z3vecN4sycl3_V13vecIiLi16EEE({{.*}})
void vec(sycl::vec<int, 16>) {}
#endif

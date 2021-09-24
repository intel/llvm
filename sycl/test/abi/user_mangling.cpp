// clang-format off
// RUN: %clangxx -fsycl -c -emit-llvm -S -o - %s | FileCheck %s --check-prefix CHK-HOST
// RUN: %clangxx -fsycl -fsycl-device-only -O0 -c -emit-llvm -S -o - %s | FileCheck %s --check-prefix CHK-DEVICE
// REQUIRES: linux
// UNSUPPORTED: libcxx

#include <CL/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
// CHK-DEVICE: define dso_local spir_func void @_Z4accdN{{.*}}8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void accd(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::device>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN{{.*}}8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN{{.*}}8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::local>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z3accN{{.*}}8accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2017ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
SYCL_EXTERNAL void acc(sycl::accessor<sycl::cl_int4, 1, sycl::access::mode::read, sycl::access::target::image>) {} 

// CHK-DEVICE: define dso_local spir_func void @_Z11private_memN{{.*}}14private_memoryIiLi1EEE({{.*}})
SYCL_EXTERNAL void private_mem(sycl::private_memory<int, 1>) {};

// CHK-DEVICE: define dso_local spir_func void @_Z5rangeN{{.*}}5rangeILi1EEE({{.*}})
SYCL_EXTERNAL void range(sycl::range<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z2idN{{.*}}2idILi1EEE({{.*}})
SYCL_EXTERNAL void id(sycl::id<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z4itemN{{.*}}2idILi1EEE({{.*}})
SYCL_EXTERNAL void item(sycl::id<1>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z3vecN{{.*}}3vecIiLi16EEE({{.*}})
SYCL_EXTERNAL void vec(sycl::vec<int, 16>) {}

// CHK-DEVICE: define dso_local spir_func void @_Z6streamN{{.*}}6streamE({{.*}})
SYCL_EXTERNAL void stream(sycl::stream) {}

// CHK-DEVICE: define dso_local spir_func void @_Z7samplerN{{.*}}7samplerE({{.*}})
SYCL_EXTERNAL void sampler(sycl::sampler) {}
#else
// CHK-HOST: define dso_local void @_Z3accN{{.*}}8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2018ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer>) {} 

// CHK-HOST: define dso_local void @_Z3accN{{.*}}8accessorIiLi1ELNS0_6access4modeE1024ELNS2_6targetE2016ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
void acc(sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::local>) {} 

// CHK-HOST: define dso_local void @_Z3accN{{.*}}8accessorINS0_3vecIiLi4EEELi1ELNS0_6access4modeE1024ELNS4_6targetE2019ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE({{.*}})
void acc(sycl::accessor<sycl::cl_int4, 1, sycl::access::mode::read, sycl::access::target::host_image>) {} 

// CHK-HOST: define dso_local void @_Z3bufN{{.*}}6bufferIiLi1ENS0_6detail17aligned_allocatorIcEEvEE({{.*}})
void buf(sycl::buffer<int>) {}

// CHK-HOST: define dso_local void @_Z3ctxN{{.*}}7contextE({{.*}})
void ctx(sycl::context) {}

// CHK-HOST: define dso_local void @_Z6deviceN{{.*}}6deviceE({{.*}})
void device(sycl::device) {}

// CHK-HOST: define dso_local void @_Z10device_evtN{{.*}}12device_eventE({{.*}})
void device_evt(sycl::device_event) {}

// CHK-HOST: define dso_local void @_Z5eventN{{.*}}5eventE({{.*}})
void event(sycl::event) {}

// CHK-HOST: define dso_local void @_Z15device_selectorRN{{.*}}15device_selectorE({{.*}})
void device_selector(sycl::device_selector&) {}

// CHK-HOST: define dso_local void @_Z7handlerRN{{.*}}7handlerE({{.*}})
void handler(sycl::handler&) {}

// CHK-HOST: define dso_local void @_Z5imageN{{.*}}5imageILi1ENS0_6detail17aligned_allocatorIhEEEE({{.*}})
void image(sycl::image<1>) {}

// CHK-HOST: define dso_local void @_Z5rangeN{{.*}}5rangeILi1EEE({{.*}})
void range(sycl::range<1>) {}

// CHK-HOST: define dso_local void @_Z2idN{{.*}}2idILi1EEE({{.*}})
void id(sycl::id<1>) {}

// CHK-HOST: define dso_local void @_Z4itemN{{.*}}4itemILi1ELb1EEE({{.*}})
void item(sycl::item<1>) {}

// CHK-HOST: define dso_local void @_Z6streamN{{.*}}6streamE({{.*}})
void stream(sycl::stream) {}

// CHK-HOST: define dso_local void @_Z7samplerN{{.*}}7samplerE({{.*}})
void sampler(sycl::sampler) {}

// CHK-HOST: define dso_local void @_Z5queueN{{.*}}5queueE({{.*}})
void queue(sycl::queue) {}

// CHK-HOST: define dso_local void @_Z7programN{{.*}}7programE({{.*}})
void program(sycl::program) {}

// CHK-HOST: define dso_local void @_Z6kernelN{{.*}}6kernelE({{.*}})
void kernel(sycl::kernel) {}

// CHK-HOST: define dso_local void @_Z8platformN{{.*}}8platformE({{.*}})
void platform(sycl::platform) {}

// CHK-HOST: define dso_local void @_Z3vecN{{.*}}3vecIiLi16EEE({{.*}})
void vec(sycl::vec<int, 16>) {}
#endif

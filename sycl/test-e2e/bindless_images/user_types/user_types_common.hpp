// This file includes common definitions and functions that are shared between
// multiple tests that use user-defined types

#include <iostream>
#include <sycl/sycl.hpp>

void printTestName(std::string name) {
#ifdef VERBOSE_PRINT
  std::cout << name << std::endl;
#endif
}

// Definitions of some typical user-defined types when sycl::vec is not used
struct my_float4 {
  my_float4() : x(0), y(0), z(0), w(0){};
  void set_all(float val) {
    x = val;
    y = val;
    z = val;
    w = val;
  }
  my_float4 &operator+=(const my_float4 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
    return *this;
  }
  float x, y, z, w;
};

struct my_float2 {
  my_float2() : x(0), y(0){};
  void set_all(float val) {
    x = val;
    y = val;
  }
  my_float2 &operator+=(const my_float2 &rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }
  float x, y;
};

struct my_half4 {
  my_half4() : x(0.f), y(0.f), z(0.f), w(0.f){};
  void set_all(float val) {
    x = val;
    y = val;
    z = val;
    w = val;
  }
  my_half4 &operator+=(const my_half4 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
    return *this;
  }
  sycl::half x, y, z, w;
};

struct my_half2 {
  my_half2() : x(0.f), y(0.f){};
  void set_all(float val) {
    x = val;
    y = val;
  }
  my_half2 &operator+=(const my_half2 &rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }
  sycl::half x, y;
};

struct my_uint4 {
  my_uint4() : x(0), y(0), z(0), w(0){};
  void set_all(uint32_t val) {
    x = val;
    y = val;
    z = val;
    w = val;
  }
  my_uint4 &operator+=(const my_uint4 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
    return *this;
  }
  uint32_t x, y, z, w;
};

struct my_uint2 {
  my_uint2() : x(0), y(0){};
  void set_all(uint32_t val) {
    x = val;
    y = val;
  }
  my_uint2 &operator+=(const my_uint2 &rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }
  uint32_t x, y;
};

struct my_ushort4 {
  my_ushort4() : x(0), y(0), z(0), w(0){};
  void set_all(uint16_t val) {
    x = val;
    y = val;
    z = val;
    w = val;
  }
  my_ushort4 &operator+=(const my_ushort4 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
    return *this;
  }
  uint16_t x, y, z, w;
};

struct my_ushort2 {
  my_ushort2() : x(0), y(0){};
  void set_all(uint16_t val) {
    x = val;
    y = val;
  }
  my_ushort2 &operator+=(const my_ushort2 &rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }
  uint16_t x, y;
};

struct my_uchar4 {
  my_uchar4() : x(0), y(0), z(0), w(0){};
  void set_all(uint8_t val) {
    x = val;
    y = val;
    z = val;
    w = val;
  }
  my_uchar4 &operator+=(const my_uchar4 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
    return *this;
  }
  uint8_t x, y, z, w;
};

struct my_uchar2 {
  my_uchar2() : x(0), y(0){};
  void set_all(uint8_t val) {
    x = val;
    y = val;
  }
  my_uchar2 &operator+=(const my_uchar2 &rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }
  uint8_t x, y;
};

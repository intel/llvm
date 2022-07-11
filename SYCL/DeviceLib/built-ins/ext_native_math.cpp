// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// OpenCL CPU driver does not support cl_khr_fp16 extension for this reason this
// test is compiled with the -fsycl-device-code-split flag

#include <cassert>
#include <sycl/sycl.hpp>

template <typename T> void assert_out_of_bound(T val, T lower, T upper) {
  assert(sycl::all(lower < val && val < upper));
}

template <>
void assert_out_of_bound<float>(float val, float lower, float upper) {
  assert(lower < val && val < upper);
}

template <>
void assert_out_of_bound<sycl::half>(sycl::half val, sycl::half lower,
                                     sycl::half upper) {
  assert(lower < val && val < upper);
}

template <typename T>
void native_tanh_tester(sycl::queue q, T val, T up, T lo) {
  T r = val;

#ifdef SYCL_EXT_ONEAPI_NATIVE_MATH
  {
    sycl::buffer<T, 1> BufR(&r, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto AccR = BufR.template get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        AccR[0] = sycl::ext::oneapi::experimental::native::tanh(AccR[0]);
      });
    });
  }

  assert_out_of_bound(r, up, lo);
#else
  assert(!"SYCL_EXT_ONEAPI_NATIVE_MATH not supported");
#endif
}

template <typename T>
void native_exp2_tester(sycl::queue q, T val, T up, T lo) {
  T r = val;

#ifdef SYCL_EXT_ONEAPI_NATIVE_MATH
  {
    sycl::buffer<T, 1> BufR(&r, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto AccR = BufR.template get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        AccR[0] = sycl::ext::oneapi::experimental::native::exp2(AccR[0]);
      });
    });
  }

  assert_out_of_bound(r, up, lo);
#else
  assert(!"SYCL_EXT_ONEAPI_NATIVE_MATH not supported");
#endif
}

int main() {

  sycl::queue q;

  const double tv[16] = {-2.0, -1.5, -1.0, 0.0, 2.0,  1.5, 1.0,   0.0,
                         -1.7, 1.7,  -1.2, 1.2, -3.0, 3.0, -10.0, 10.0};
  const double tl[16] = {-0.97, -0.91, -0.77, -0.1, 0.95, 0.89, 0.75,  -0.1,
                         -0.94, 0.92,  -0.84, 0.82, -1.0, 0.98, -1.10, 0.98};
  const double tu[16] = {-0.95, -0.89, -0.75, 0.1,  0.97,  0.91, 0.77,  0.1,
                         -0.92, 0.94,  -0.82, 0.84, -0.98, 1.00, -0.98, 1.10};

  native_tanh_tester<float>(q, tv[0], tl[0], tu[0]);
  native_tanh_tester<sycl::float2>(q, {tv[0], tv[1]}, {tl[0], tl[1]},
                                   {tu[0], tu[1]});
  native_tanh_tester<sycl::float3>(
      q, {tv[0], tv[1], tv[2]}, {tl[0], tl[1], tl[2]}, {tu[0], tu[1], tu[2]});
  native_tanh_tester<sycl::float4>(q, {tv[0], tv[1], tv[2], tv[3]},
                                   {tl[0], tl[1], tl[2], tl[3]},
                                   {tu[0], tu[1], tu[2], tu[3]});
  native_tanh_tester<sycl::float8>(
      q, {tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7]},
      {tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7]},
      {tu[0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6], tu[7]});
  native_tanh_tester<sycl::float16>(
      q,
      {tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7], tv[8], tv[9],
       tv[10], tv[11], tv[12], tv[13], tv[14], tv[15]},
      {tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7], tl[8], tl[9],
       tl[10], tl[11], tl[12], tl[13], tl[14], tl[15]},
      {tu[0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6], tu[7], tu[8], tu[9],
       tu[10], tu[11], tu[12], tu[13], tu[14], tu[15]});

  if (q.get_device().has(sycl::aspect::fp16)) {

    native_tanh_tester<sycl::half>(q, tv[0], tl[0], tu[0]);
    native_tanh_tester<sycl::half2>(q, {tv[0], tv[1]}, {tl[0], tl[1]},
                                    {tu[0], tu[1]});
    native_tanh_tester<sycl::half3>(
        q, {tv[0], tv[1], tv[2]}, {tl[0], tl[1], tl[2]}, {tu[0], tu[1], tu[2]});
    native_tanh_tester<sycl::half4>(q, {tv[0], tv[1], tv[2], tv[3]},
                                    {tl[0], tl[1], tl[2], tl[3]},
                                    {tu[0], tu[1], tu[2], tu[3]});
    native_tanh_tester<sycl::half8>(
        q, {tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7]},
        {tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7]},
        {tu[0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6], tu[7]});
    native_tanh_tester<sycl::half16>(
        q,
        {tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7], tv[8], tv[9],
         tv[10], tv[11], tv[12], tv[13], tv[14], tv[15]},
        {tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7], tl[8], tl[9],
         tl[10], tl[11], tl[12], tl[13], tl[14], tl[15]},
        {tu[0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6], tu[7], tu[8], tu[9],
         tu[10], tu[11], tu[12], tu[13], tu[14], tu[15]});

    const double ev[16] = {-2.0, -1.5, -1.0, 0.0, 2.0, 1.5, 1.0, 0.0,
                           -2.0, -1.5, -1.0, 0.0, 2.0, 1.5, 1.0, 0.0};
    const double el[16] = {0.1, 0.34, 0.4, -0.9, 3.9, 2.7, 1.9, -0.9,
                           0.1, 0.34, 0.4, -0.9, 3.9, 2.7, 1.9, -0.9};
    const double eu[16] = {0.3, 0.36, 0.6, 1.1, 4.1, 2.9, 2.1, 1.1,
                           0.3, 0.36, 0.6, 1.1, 4.1, 2.9, 2.1, 1.1};

    native_exp2_tester<sycl::half>(q, ev[0], el[0], eu[0]);
    native_exp2_tester<sycl::half2>(q, {ev[0], ev[1]}, {el[0], el[1]},
                                    {eu[0], eu[1]});
    native_exp2_tester<sycl::half3>(
        q, {ev[0], ev[1], ev[2]}, {el[0], el[1], el[2]}, {eu[0], eu[1], eu[2]});
    native_exp2_tester<sycl::half4>(q, {ev[0], ev[1], ev[2], ev[3]},
                                    {el[0], el[1], el[2], el[3]},
                                    {eu[0], eu[1], eu[2], eu[3]});
    native_exp2_tester<sycl::half8>(
        q, {ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7]},
        {el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]},
        {eu[0], eu[1], eu[2], eu[3], eu[4], eu[5], eu[6], eu[7]});
    native_exp2_tester<sycl::half16>(
        q,
        {ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8], ev[9],
         ev[10], ev[11], ev[12], ev[13], ev[14], ev[15]},
        {el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7], el[8], el[9],
         el[10], el[11], el[12], el[13], el[14], el[15]},
        {eu[0], eu[1], eu[2], eu[3], eu[4], eu[5], eu[6], eu[7], eu[8], eu[9],
         eu[10], eu[11], eu[12], eu[13], eu[14], eu[15]});
  }

  return 0;
}

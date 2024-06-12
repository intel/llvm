// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "esimd_test_utils.hpp"

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr unsigned Size = 4;
constexpr unsigned VL = 4;
constexpr unsigned GroupSize = 1;

bool isFlagSet(intelex::fp_mode fp_control, intelex::fp_mode flag) {
  return (static_cast<uint32_t>(fp_control) & static_cast<uint32_t>(flag)) ==
         static_cast<uint32_t>(flag);
}

std::string fp_control_to_sring(intelex::fp_mode fp_control) {
  std::string Modes;
  if (isFlagSet(fp_control, intelex::fp_mode::round_toward_zero)) {
    Modes += " round_toward_zero";
  } else if (isFlagSet(fp_control, intelex::fp_mode::round_upward)) {
    Modes += " round_upward";
  } else if (isFlagSet(fp_control, intelex::fp_mode::round_downward)) {
    Modes += " round_downward";
  } else if (isFlagSet(fp_control, intelex::fp_mode::round_to_nearest)) {
    Modes += " round_to_nearest";
  }

  if (isFlagSet(fp_control, intelex::fp_mode::denorm_allow)) {
    Modes += " denorm_allow";
  } else if (isFlagSet(fp_control, intelex::fp_mode::denorm_f_allow)) {
    Modes += " denorm_f_allow";
  } else if (isFlagSet(fp_control, intelex::fp_mode::denorm_d_allow)) {
    Modes += " denorm_d_allow";
  } else if (isFlagSet(fp_control, intelex::fp_mode::denorm_hf_allow)) {
    Modes += " denorm_hf_allow";
  } else if (isFlagSet(fp_control, intelex::fp_mode::denorm_ftz)) {
    Modes += " denorm_ftz";
  }

  return Modes;
}

template <intelex::fp_mode, class T> class TestKernel;

enum result { eq, lt, gt };

template <intelex::fp_mode fp_mode, class T>
std::pair<result, result> test_fp_control(queue &q, T ValA, T ValB) {
  std::cout << "Testing fp_control:" << fp_control_to_sring(fp_mode)
            << " for type " << typeid(T).name() << std::endl;
  std::cout << "ValA:  " << std::setprecision(10) << ValA << " ValB: " << ValB
            << std::endl;

  T *A = malloc_shared<T>(Size, q);
  T *B = malloc_shared<T>(Size, q);
  T *C = malloc_shared<T>(Size, q);
  T *D = malloc_shared<T>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = ValA;
    B[i] = ValB;
  }

  // We need that many work items. Each processes VL elements of data.
  range<1> GlobalRange{Size / VL};
  // Number of workitems in each workgroup.
  range<1> LocalRange{GroupSize};

  nd_range<1> Range(GlobalRange, LocalRange);

  syclex::properties properties{intelex::fp_control<fp_mode>};
  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class TestKernel<fp_mode, T>>(
        Range, properties, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          using namespace sycl::ext::intel::esimd;

          int i = ndi.get_global_id(0);
          simd<T, VL> va;
          va.copy_from(A + i * VL);
          simd<T, VL> vb;
          vb.copy_from(B + i * VL);
          simd<T, VL> vc = va + vb;
          simd<T, VL> vd = va - vb;
          vc.copy_to(C + i * VL);
          vd.copy_to(D + i * VL);
        });
  });
  e.wait();

  std::cout << "Addition: ";
  for (unsigned i = 0; i < Size; ++i)
    std::cout << std::scientific << std::setprecision(30)
              << (i != 0 ? "; " : "") << C[i];

  std::cout << std::endl;
  std::cout << "Substraction: ";
  for (unsigned i = 0; i < Size; ++i)
    std::cout << std::scientific << std::setprecision(30)
              << (i != 0 ? "; " : "") << D[i];

  std::cout << std::endl << std::endl;

  auto res_add =
      C[0] > ValA ? result::gt : (C[0] < ValA ? result::lt : result::eq);
  auto res_sub =
      D[0] > ValA ? result::gt : (D[0] < ValA ? result::lt : result::eq);

  free(A, q);
  free(B, q);
  free(C, q);
  free(D, q);

  return {res_add, res_sub};
}

template <typename T, intelex::fp_mode DenormMode>
void test_rounding_mode_and_denormals(queue &q, T Base, T Epsilon, T Denorm) {
  // Test rounding modes.
  {
    auto [res_add1, res_sub1] =
        test_fp_control<intelex::fp_mode::round_to_nearest, T>(q, Base,
                                                               Epsilon);
    assert(res_add1 == result::eq && res_sub1 == result::eq);
    auto [res_add2, res_sub2] =
        test_fp_control<intelex::fp_mode::round_upward, T>(q, Base, Epsilon);
    assert(res_add2 == result::gt && res_sub2 == result::eq);
    auto [res_add3, res_sub3] =
        test_fp_control<intelex::fp_mode::round_downward, T>(q, Base, Epsilon);
    assert(res_add3 == result::eq && res_sub3 == result::lt);
    auto [res_add4, res_sub4] =
        test_fp_control<intelex::fp_mode::round_toward_zero, T>(q, Base,
                                                                Epsilon);
    assert(res_add4 == result::eq && res_sub4 == result::lt);
  }

  // Test that denormals are flushed to zero by default and if denorm_ftz is
  // specified explicitely.
  {
    auto [res_add1, res_sub1] =
        test_fp_control<intelex::fp_mode::round_to_nearest, T>(q, Base, Denorm);
    assert(res_add1 == result::eq && res_sub1 == result::eq);
    auto [res_add2, res_sub2] =
        test_fp_control<intelex::fp_mode::round_upward, T>(q, Base, Denorm);
    assert(res_add2 == result::eq && res_sub2 == result::eq);
    auto [res_add3, res_sub3] = test_fp_control<
        intelex::fp_mode::round_downward | intelex::fp_mode::denorm_ftz, T>(
        q, Base, Denorm);
    assert(res_add3 == result::eq && res_sub3 == result::eq);
    auto [res_add4, res_sub4] = test_fp_control<
        intelex::fp_mode::round_toward_zero | intelex::fp_mode::denorm_ftz, T>(
        q, Base, Denorm);
    assert(res_add3 == result::eq && res_sub3 == result::eq);
  }

  // Test the mode when denormals are allowed for type T.
  {
    auto [res_add1, res_sub1] =
        test_fp_control<intelex::fp_mode::round_to_nearest | DenormMode, T>(
            q, Base, Denorm);
    assert(res_add1 == result::eq && res_sub1 == result::eq);
    auto [res_add2, res_sub2] =
        test_fp_control<intelex::fp_mode::round_upward | DenormMode, T>(q, Base,
                                                                        Denorm);
    assert(res_add2 == result::gt && res_sub2 == result::eq);
    auto [res_add3, res_sub3] =
        test_fp_control<intelex::fp_mode::round_downward | DenormMode, T>(
            q, Base, Denorm);
    assert(res_add3 == result::eq && res_sub3 == result::lt);
    auto [res_add4, res_sub4] =
        test_fp_control<intelex::fp_mode::round_toward_zero | DenormMode, T>(
            q, Base, Denorm);
    assert(res_add4 == result::eq && res_sub4 == result::lt);
  }
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  // Test rounding modes and denormals for float.
  test_rounding_mode_and_denormals<float, intelex::fp_mode::denorm_f_allow>(
      q, 1.0f, 1e-23f, 1e-38f);

  // Test rounding modes and denormals for half.
  if (dev.has(aspect::fp16))
    test_rounding_mode_and_denormals<sycl::half,
                                     intelex::fp_mode::denorm_hf_allow>(
        q, 1.0f, 1e-4f, 1e-6f);

  // Test rounding modes and denormals for double.
  if (dev.has(aspect::fp64))
    test_rounding_mode_and_denormals<double, intelex::fp_mode::denorm_d_allow>(
        q, 1.0, 1e-128, 1e-308);

  return 0;
}

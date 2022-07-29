#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

class MyInt32Const;
class MyFloatConst;
class MyConst;

using namespace sycl;

class KernelAAAi;
class KernelBBBf;

int global_val = 10;

// Fetch a value at runtime.
int get_value() { return global_val; }

float foo(
    const sycl::ext::oneapi::experimental::spec_constant<float, MyFloatConst>
        &f32) {
  return f32;
}

struct SCWrapper {
  SCWrapper(sycl::program &p)
      : SC1(p.set_spec_constant<class sc_name1, int>(4)),
        SC2(p.set_spec_constant<class sc_name2, int>(2)) {}

  sycl::ext::oneapi::experimental::spec_constant<int, class sc_name1> SC1;
  sycl::ext::oneapi::experimental::spec_constant<int, class sc_name2> SC2;
};

// MyKernel is used to test default constructor
using AccT = sycl::accessor<int, 1, sycl::access::mode::write>;
using ScT = sycl::ext::oneapi::experimental::spec_constant<int, MyConst>;

struct MyKernel {
  MyKernel(AccT &Acc) : Acc(Acc) {}

  void setConst(ScT Sc) { this->Sc = Sc; }

  void operator()() const { Acc[0] = Sc.get(); }
  AccT Acc;
  ScT Sc;
};

int main(int argc, char **argv) {
  global_val = argc + 16;

  sycl::queue q(default_selector{}, [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << e0.what();
      } catch (std::exception &e1) {
        std::cout << e1.what();
      } catch (...) {
        std::cout << "*** catch (...)\n";
      }
    }
  });

  std::cout << "Running on " << q.get_device().get_info<info::device::name>()
            << "\n";
  std::cout << "global_val = " << global_val << "\n";
  sycl::program program1(q.get_context());
  sycl::program program2(q.get_context());
  sycl::program program3(q.get_context());
  sycl::program program4(q.get_context());

  const int goldi = get_value();
  // TODO make this floating point once supported by the compiler
  const float goldf = get_value();

  sycl::ext::oneapi::experimental::spec_constant<int32_t, MyInt32Const> i32 =
      program1.set_spec_constant<MyInt32Const>(goldi);

  sycl::ext::oneapi::experimental::spec_constant<float, MyFloatConst> f32 =
      program2.set_spec_constant<MyFloatConst>(goldf);

  sycl::ext::oneapi::experimental::spec_constant<int, MyConst> sc =
      program4.set_spec_constant<MyConst>(goldi);

  program1.build_with_kernel_type<KernelAAAi>();
  // Use an option (does not matter which exactly) to test different internal
  // SYCL RT execution path
  program2.build_with_kernel_type<KernelBBBf>("-cl-fast-relaxed-math");

  SCWrapper W(program3);
  program3.build_with_kernel_type<class KernelWrappedSC>();

  program4.build_with_kernel_type<MyKernel>();

  int goldw = 6;

  std::vector<int> veci(1);
  std::vector<float> vecf(1);
  std::vector<int> vecw(1);
  std::vector<int> vec(1);
  try {
    sycl::buffer<int, 1> bufi(veci.data(), veci.size());
    sycl::buffer<float, 1> buff(vecf.data(), vecf.size());
    sycl::buffer<int, 1> bufw(vecw.data(), vecw.size());
    sycl::buffer<int, 1> buf(vec.data(), vec.size());

    q.submit([&](sycl::handler &cgh) {
      auto acci = bufi.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<KernelAAAi>(program1.get_kernel<KernelAAAi>(),
                                  [=]() { acci[0] = i32.get(); });
    });
    q.submit([&](sycl::handler &cgh) {
      auto accf = buff.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<KernelBBBf>(program2.get_kernel<KernelBBBf>(),
                                  [=]() { accf[0] = foo(f32); });
    });

    q.submit([&](sycl::handler &cgh) {
      auto accw = bufw.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<KernelWrappedSC>(
          program3.get_kernel<KernelWrappedSC>(),
          [=]() { accw[0] = W.SC1.get() + W.SC2.get(); });
    });
    // Check spec_constant default construction with subsequent initialization
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      // Specialization constants specification says:
      //   sycl::experimental::spec_constant is default constructible,
      //   although the object is not considered initialized until the result of
      //   the call to sycl::program::set_spec_constant is assigned to it.
      MyKernel Kernel(acc); // default construct inside MyKernel instance
      Kernel.setConst(sc);  // initialize to sc, returned by set_spec_constant

      cgh.single_task<MyKernel>(program4.get_kernel<MyKernel>(), Kernel);
    });

  } catch (sycl::exception &e) {
    std::cout << "*** Exception caught: " << e.what() << "\n";
    return 1;
  }
  bool passed = true;
  int vali = veci[0];

  if (vali != goldi) {
    std::cout << "*** ERROR: " << vali << " != " << goldi << "(gold)\n";
    passed = false;
  }

  if (std::fabs(vecf[0] - goldf) > std::numeric_limits<float>::epsilon()) {
    std::cout << "*** ERROR: " << vecf[0] << " != " << goldf << "(gold)\n";
    passed = false;
  }
  int valw = vecw[0];

  if (valw != goldw) {
    std::cout << "*** ERROR: " << valw << " != " << goldw << "(gold)\n";
    passed = false;
  }
  int val = vec[0];

  if (val != goldi) {
    std::cout << "*** ERROR: " << val << " != " << goldi << "(gold)\n";
    passed = false;
  }
  std::cout << (passed ? "passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}

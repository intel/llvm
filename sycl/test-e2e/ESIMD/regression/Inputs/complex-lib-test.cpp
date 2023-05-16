#include <cstdio>
#include <sycl/sycl.hpp>

sycl::event iota(size_t n, sycl::buffer<int, 1> &d, sycl::queue &Q);
sycl::event add(size_t n, sycl::buffer<int, 1> &buf_a,
                sycl::buffer<int, 1> &buf_b, sycl::buffer<int, 1> &buf_c,
                sycl::queue &Q);

int main(int argc, char *argv[]) {
  try {
    size_t i;
    size_t N = 1024;
    sycl::device D(sycl::default_selector_v);
    sycl::context Ctx(D);
    sycl::queue Q(Ctx, D);

    std::vector<int> A(N), B(N), C(N);
    {
      sycl::buffer<int, 1> buf_A(A.data(), N);
      sycl::buffer<int, 1> buf_B(B.data(), N);
      iota(N, buf_A, Q);
      iota(N, buf_B, Q);
    }

    bool pass = true;
    for (i = 0; i < 10; ++i) {
      pass = pass && (A[i] == i);
      pass = pass && (B[i] == i);
    }

    {
      sycl::buffer<int, 1> buf_A(A.data(), N);
      sycl::buffer<int, 1> buf_B(B.data(), N);
      sycl::buffer<int, 1> buf_C(C.data(), N);
      add(N, buf_A, buf_B, buf_C, Q);
    }

    for (i = 0; i < 10; ++i) {
      pass = pass && (A[i] + B[i] == C[i]);
    }

    fprintf(stdout, "%s: %s\n", argv[0], pass ? "PASS" : "FAIL");
  } catch (sycl::exception const &se) {
    fprintf(stderr, "%s failed with %s (%d)\n", argv[0], se.what(),
            se.code().value());

    return 1;
  } catch (std::exception const &e) {
    fprintf(stderr, "failed with %s\n", e.what());
    return 2;
  } catch (...) {
    fprintf(stderr, "failed\n");
    return -1;
  }
  return 0;
}

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/usm.hpp>

// Sanity test that checks to see if idiomatic code involving work_group_memory
// objects compiles and runs with no errors.

namespace syclex = sycl::ext::oneapi::experimental;
sycl::queue global_q;

constexpr size_t SIZE = 4096;
constexpr size_t WGSIZE = 256;

struct point {
  int x;
  int y;
};

void simple_inc(const syclex::work_group_memory<int> &mem) { mem++; }

void fancy_inc(syclex::work_group_memory<int> mem) {
  syclex::work_group_memory<int> t = mem;
  t = mem;
  t++;
}

void test_breadth() {
  sycl::queue q;
  global_q = q;

  int *res = sycl::malloc_host<int>(16, q);

  q.submit([&](sycl::handler &cgh) {
     syclex::work_group_memory<int> mem1{cgh};
     syclex::work_group_memory<int[10]> mem2{cgh};
     syclex::work_group_memory<int[10]> mem3{cgh};
     syclex::work_group_memory<int[]> mem4{5, cgh};
     syclex::work_group_memory<int[][10]> mem5{2, cgh};
     syclex::work_group_memory<int[][10]> mem6{2, cgh};
     syclex::work_group_memory<point> mem7{cgh};
     syclex::work_group_memory<point[][10]> mem8{2, cgh};

     cgh.single_task([=] {
       // Operations on scalar
       ++mem1;
       mem1++;
       mem1 += 1;
       mem1 = mem1 + 1;
       int *p1 = &mem1;
       (*p1)++;
       simple_inc(mem1);
       fancy_inc(mem1);
       res[0] = *(mem1.get_multi_ptr());
       res[1] = mem1;

       // Operations on bounded array
       mem2[4] = mem2[4] + 1;
       int(*p2)[10] = &mem2;
       (*p2)[4]++;
       res[2] = mem2.get_multi_ptr()[4];
       res[3] = mem2[4];

       mem3[4] = mem3[4] + 1;
       int(*p3)[10] = &mem3;
       (*p3)[4]++;
       res[4] = mem3.get_multi_ptr()[4];
       res[5] = mem3[4];

       // Operations on unbounded array
       mem4[4] = mem4[4] + 1;
       int(*p4)[] = &mem4;
       (*p4)[4]++;
       res[6] = mem4.get_multi_ptr()[4];
       res[7] = mem4[4];

       // Operations on unbounded multi-dimensional array
       mem5[1][5] = mem5[1][5] + 1;
       mem5[1][7] = mem5[1][7] + 1;
       res[8] = mem5.get_multi_ptr()[10 + 5];
       res[9] = mem5[1][7];

       mem6[1][5] = mem6[1][5] + 1;
       mem6[1][7] = mem6[1][7] + 1;
       res[10] = mem6.get_multi_ptr()[10 + 5];
       res[11] = mem6[1][7];

       // Operations on scalar struct
       (&mem7)->x++;
       (&mem7)->y += 1;
       point pnt = mem7;
       pnt.x++;
       pnt.y++;
       mem7 = pnt;
       res[12] = (&mem7)->x;
       res[13] = (&mem7)->y;

       // Operations on unbounded multi-dimensional array of struct
       mem8[1][5].x++;
       mem8[1][5].y += 1;
       res[14] = mem8.get_multi_ptr()[10 + 5].x;
       res[15] = mem8[1][5].y;
     });
   }).wait();
}

void test_basic() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
     // Allocate one element for each work-item in the work-group.
     syclex::work_group_memory<int[WGSIZE]> mem{cgh};

     sycl::nd_range ndr{{SIZE}, {WGSIZE}};
     cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
       size_t id = it.get_local_linear_id();

       // Each work-item has its own dedicated element of the array.
       mem[id] = 0;
     });
   }).wait();
}

void test_operations() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
     syclex::work_group_memory<int> mem1{cgh};      // scalar
     syclex::work_group_memory<int[10]> mem2{cgh};  // bounded array
     syclex::work_group_memory<int[]> mem3{5, cgh}; // unbounded array
     syclex::work_group_memory<int[][10]> mem4{2,
                                               cgh}; // multi-dimensional array
     syclex::work_group_memory<point[10]> mem5{cgh}; // array of struct

     sycl::nd_range ndr{{SIZE}, {WGSIZE}};
     cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
       if (it.get_group().leader()) {
         // A "work_group_memory" templated on a scalar type acts much like the
         // enclosed scalar type.
         ++mem1;
         mem1++;
         mem1 += 1;
         mem1 = mem1 + 1;
         int *p1 = &mem1;

         // A "work_group_memory" templated on an array type (either bounded or
         // unbounded) acts like an array.
         ++mem2[4];
         mem2[4]++;
         mem2[4] = mem2[4] + 1;
         int *p2 = &mem2[4];

         // A multi-dimensional array works as expected.
         mem4[1][5] = mem4[1][5] + 1;
         mem4[1][7] = mem4[1][7] + 1;

         // An array of structs works as expected too.
         mem5[1].x++;
         mem5[1].y = mem5[1].y + 1;
       }
     });
   }).wait();
}

int main() {
  test_breadth();
  test_basic();
  test_operations();
}

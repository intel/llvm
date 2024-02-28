// RUN: %clangxx -fsycl -fpreview-breaking-changes %s -o %t.out
#include <sycl/sycl.hpp>
#include <iostream>

  inline uint32_t fma(uint32_t a, uint32_t b, uint32_t c)
{
   sycl::half2 ah=sycl::bit_cast<sycl::half2,uint32_t>(a);
   sycl::half2 bh=sycl::bit_cast<sycl::half2,uint32_t>(b);
   sycl::half2 ch=sycl::bit_cast<sycl::half2,uint32_t>(c);
   
   auto res=sycl::fma(ah,bh,ch);
   return sycl::bit_cast<uint32_t,sycl::half2>(res);
}                                                                    

void fmakernel(uint32_t* a, uint32_t* b, uint32_t* c,uint32_t *d,
               const sycl::nd_item<3> &item_ct1)
{
    d[item_ct1.get_local_id(2)] =
        fma(a[item_ct1.get_local_id(2)], b[item_ct1.get_local_id(2)],
              c[item_ct1.get_local_id(2)]);
}





int main()
{
    sycl::queue q_ct1;
    int N=1;
    uint32_t *ha=(uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t *hb=(uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t *hc=(uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t *hd=(uint32_t*)malloc(N*sizeof(uint32_t));

    uint32_t *da,*db,*dc,*dd;
    da = sycl::malloc_device<uint32_t>(N, q_ct1);
    db = sycl::malloc_device<uint32_t>(N, q_ct1);
    dc = sycl::malloc_device<uint32_t>(N, q_ct1);
    dd = sycl::malloc_device<uint32_t>(N, q_ct1);

    for (int i=0;i<N;i++)
    {
        ha[i]=899628258;
        hb[i]=800306892;
        hc[i]=800306892;
    }

    q_ct1.memcpy(da, ha, N * sizeof(uint32_t));
    q_ct1.memcpy(db, hb, N * sizeof(uint32_t));
    q_ct1.memcpy(dc, hc, N * sizeof(uint32_t));

    {
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N), sycl::range<3>(1, 1, N)),
            [=](sycl::nd_item<3> item_ct1) {
                fmakernel(da, db, dc, dd, item_ct1);
            });
    }

    q_ct1.memcpy(hd, dd, N * sizeof(uint32_t)).wait();

    for(int i=0;i<N;i++)
    {
        std::cout<<hd[i]<<std::endl;
    }
}

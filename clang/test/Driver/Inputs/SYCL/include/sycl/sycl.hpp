#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_EXTERNAL __attribute__((sycl_device))
#else
#define SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

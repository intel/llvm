
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
int main() {  hipError_t r = hipInit(0); return r; }


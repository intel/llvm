// For testing overload-res_*.cpp

__attribute__((device)) void callee_device() {} // #callee_device
__attribute__((host)) void callee_host() {}     // #callee_host
void callee_host_implicit() {}                  // #callee_host_implicit
__attribute__((device)) __attribute__((host)) void callee_hostdevice() {
}                                               // #callee_hostdevice
constexpr void callee_hostdevice_implicit() {}  // #callee_hostdevice_implicit
__attribute__((global)) void callee_global() {} // #callee_global

// Functions to check overload resolution favours the right overload based on
// its host/device/global attribute
//
// Note: it is normally not possible to overload a function decorated with
// __device__ with another decorated with __host__ __device__. But a hole in
// checks make it possible by declaring the functions in 2 distinct namespaces
// and pulling one into the other namespace. Unfortunately, this is extensively
// used in system headers.
//
// Note 2: we don't include an overload with __global__ as it always will be
// ambiguous with __host__

namespace bar {
// Note: constexpr implies implicit host-device
inline constexpr void overload() {}
} // namespace bar

__attribute__((device)) void overload() {}
__attribute__((host)) void overload() {}

namespace bar {
// the host only and device only function in the same namespace as the
// host-device one.
using ::overload;
} // namespace bar

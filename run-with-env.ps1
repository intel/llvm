$env:SYCL_UR_TRACE = 1
$env:ONEAPI_DEVICE_SELECTOR = "opencl:gpu"

.\build-d\bin\pi_release.cpp.tmp.out.exe

Remove-Item env:SYCL_UR_TRACE
Remove-Item env:ONEAPI_DEVICE_SELECTOR

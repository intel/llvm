# commit 70053658b3f741f41cf9b62e063e8c4fe51957e0
# Author: Przemek Malon <przemek.malon@codeplay.com>
# Date:   Wed Jan 8 19:53:17 2025 +0000
# Enable creation of images backed by host USM
#
#   Small patch to enable bindless images backed by host USM in the CUDA
#   adapter.
#
#   Host and Device USM pointers are usable across the host and device
#   for all versions of CUDA that we support. There is no need to provide
#   the `CU_MEMHOSTALLOC_DEVICEMAP` flag during allocation, or calling
#   `cuMemHostGetDevicePointer` to retrieve a device usable address.
#
#   Passing a `CU_MEMHOSTALLOC_WRITECOMBINED` flag to the host USM
#   allocation will enhance performance in certain scenarios, however, an
#   extension allowing this is not yet available.
set(UNIFIED_RUNTIME_TAG 70053658b3f741f41cf9b62e063e8c4fe51957e0)

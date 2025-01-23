# commit eeff9f4a6e0ed51ba459fd923724fb4c3dd545d7
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
set(UNIFIED_RUNTIME_TAG eeff9f4a6e0ed51ba459fd923724fb4c3dd545d7)

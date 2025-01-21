macro(find_cuda_cupti_library)
# The following if can be removed when FindCUDA -> FindCUDAToolkit
  find_library(CUDA_cupti_LIBRARY
    NAMES cupti
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
          ENV CUDA_PATH
    PATH_SUFFIXES nvidia/current lib64 lib/x64 lib
                  ../extras/CUPTI/lib64/
                  ../extras/CUPTI/lib/
  )
endmacro()

macro(find_cuda_cupti_include_dir)
  find_path(CUDA_CUPTI_INCLUDE_DIR cupti.h PATHS
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
      "${CUDA_INCLUDE_DIRS}/../extras/CUPTI/include"
      "${CUDA_INCLUDE_DIRS}"
      NO_DEFAULT_PATH)
endmacro()


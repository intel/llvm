import json

def parse_dependencies(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    linux_dependencies = data.get('linux', {})
    keys_to_parse = ['oclcpu', 'tbb', 'compute_runtime', 'fpgaemu'] #, 'fpga']
    parsed_dependencies = {}

    for key in keys_to_parse:
        if key in linux_dependencies:
            value = linux_dependencies[key]
            root = value.get('root').replace('{DEPS_ROOT}', '$DEPS') if value.get('root') else None
            version = value.get('version')
            parsed_dependencies[key] = {'root': root, 'version': version}
    return parsed_dependencies

parsed_deps = parse_dependencies('devops/dependencies.json')

# ARCHIVE_ROOT="/nfs/site/proj/icl/xarch/archive"
# fpga_full         = parsed_deps['fpga']['root']

# Full
ocl_fpga_emu_full = parsed_deps['fpgaemu']['root'] + '/' + parsed_deps['fpgaemu']['version']
ocl_gpu_rt_full   = parsed_deps['compute_runtime']['root'] + '/' + parsed_deps['compute_runtime']['version']
ocl_cpu_rt_full   = parsed_deps['oclcpu']['root'] + '/' + parsed_deps['oclcpu']['version']
tbb_full          = parsed_deps['tbb']['root'] #+ '/' + parsed_deps['tbb']['version']

# For debug purposes
# print(ocl_fpga_emu_full)
# print(ocl_gpu_rt_full)
# print(ocl_cpu_rt_full)
# print(tbb_full)

print("export DEPS=/rdrive/ref")
print("export DPCPP_HOME=$(cd ..; pwd)")
# print(f"export PATH=$DPCPP_HOME/llvm/build/bin:{ocl_fpga_emu_full}/bin:{fpga_full}/build/bin:{ocl_gpu_rt_full}:$PATH")
print(f"export PATH=$DPCPP_HOME/llvm/build/bin:{ocl_fpga_emu_full}/bin:{ocl_gpu_rt_full}:$PATH")
print(f"export OCL_ICD_FILENAMES={ocl_fpga_emu_full}/libintelocl_emu.so:libalteracl.so:{ocl_cpu_rt_full}/libintelocl.so:{ocl_gpu_rt_full}/libigdrcl.so")
print(f"export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:{ocl_fpga_emu_full}:{ocl_cpu_rt_full}:{tbb_full}/2022.0.0.196/tbb/2022.0/lib:{ocl_gpu_rt_full}")

# print(f"export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:{ocl_fpga_emu_full}/intel64_lin:{ocl_fpga_emu_full}:{ocl_cpu_rt_full}/intel64_lin:{tbb_full}/lib/intel64/gcc4.8:{ocl_gpu_rt_full}")
# /rdrive/ref/tbb/lin/2022.0.0.196/tbb/2022.0/lib
# {tbb_full}/lib/intel64/gcc4.8

# /rdrive/ref/mpi/lin/2021.11.0.49490/intel64/opt/mpi/libfabric/lib:/rdrive/ref/mpi/lin/2021.11.0.49490/intel64/lib:/nfs/sc/proj/icl/cmplrarch/comp/onemkl/lin/20250209_cev_nightly/lib:/nfs/sc/proj/icl/cmplrarch/comp/onednnl/lin/20230609_8d22ee/lib:/rdrive/ref/tbb/lin/2022.0.0.196/tbb/2022.0/lib:/nfs/sc/proj/icl/cmplrarch/deploy_syclos/llorgefi2linux/20250220_160000/build/linux_qa_release/lib:/rdrive/ref/gcc/14.2.0/ubt2204/efi2/lib64:/rdrive/ref/gcc/14.2.0/ubt2204/efi2/lib32:/rdrive/ref/gcc/14.2.0/ubt2204/efi2/lib:/rdrive/ref/opencl/runtime/linux/oclfpgaemu/2024.18.10.0.08:/rdrive/ref/opencl/runtime/linux/oclcpu/2024.18.10.0.08:/nfs/sc/proj/icl/cmplrarch/comp/oclfpga/linux/20240912144354/build/host/linux64/lib:/nfs/sc/proj/icl/cmplrarch/comp/oclfpga/linux/20240912144354/build/linux64/lib:/rdrive/ref/opencl/runtime/linux/oclgpu/24.52.32224.5:/rdrive/ref/opencl/runtime/linux/oclgpu/24.52.32224.5/intel-opencl:/rdrive/ref/opencl/runtime/linux/level_zero_loader/1.19.2

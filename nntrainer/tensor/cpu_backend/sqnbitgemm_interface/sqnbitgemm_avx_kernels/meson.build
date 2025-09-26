
sqnbitgemm_avx_kernels_headers = [
    'FgemmKernelAvxCommon.h',
    'sqnbitgemm_kernel_avx_common_fp32.h',
    'sqnbitgemm_kernel_avx_common_int8.h',
    'sqnbitgemm_kernel_avx_common.h',
    'sqnbitgemm_kernel_avx2_int8_blklen16.h',
    'sqnbitgemm_kernel_avx2_int8_blklen32.h',
    'sqnbitgemm_kernel_avx2_int8_blklen64.h',
]

sqnbitgemm_avx_kernels_sources = [
    'SgemmKernelAvx.S',
    'sqnbitgemm_kernel_avx2.cpp'
]

foreach h : sqnbitgemm_avx_kernels_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


foreach s : sqnbitgemm_avx_kernels_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

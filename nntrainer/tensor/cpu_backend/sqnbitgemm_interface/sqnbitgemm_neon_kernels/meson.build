
sqnbitgemm_avx_kernels_headers = [
    'sqnbitgemm_kernel_neon.h',
]

sqnbitgemm_avx_kernels_sources = [
    'sqnbitgemm_kernel_neon.cpp',
    'sqnbitgemm_kernel_neon_int8.cpp'
]

foreach h : sqnbitgemm_avx_kernels_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


foreach s : sqnbitgemm_avx_kernels_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

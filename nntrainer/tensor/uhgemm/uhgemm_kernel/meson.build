uhgemm_kernel_headers = [
  'uhgemm_kernel.h',
]


uhgemm_kernel_sources = [
    'uhgemm_kernel_1x4.cpp',
    'uhgemm_kernel_1x8.cpp',
    'uhgemm_kernel_4x4.cpp',
    'uhgemm_kernel_4x8.cpp',
    'uhgemm_kernel_8x8.cpp',
]

# if get_option('false')
 uhgemm_kernel_sources += 'uhgemm_kernel_8x16_experimental.cpp'
# else
  # uhgemm_kernel_sources += 'uhgemm_kernel_8x16.cpp'
# endif

foreach s : uhgemm_kernel_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : uhgemm_kernel_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


gemm_headers = [
  'gemm_util.h',
]
gemm_sources = [
  'gemm_util.cpp',
]

subdir('uhgemm')
nntrainer_inc += include_directories('uhgemm')
nntrainer_inc_abs += meson.current_source_dir() / 'uhgemm'

if get_option('enable-fp16')
    subdir('hgemm')
    nntrainer_inc += include_directories('hgemm')
    nntrainer_inc_abs += meson.current_source_dir() / 'hgemm'
endif

foreach s : gemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : gemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


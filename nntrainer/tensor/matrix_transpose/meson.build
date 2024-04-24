hgemm_headers = [
  'transpose.h',
  'transpose_kernel.h',
  'transpose_kernel_8x8.h',
]

hgemm_sources = [
    'transpose.cpp'
]

foreach s : hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


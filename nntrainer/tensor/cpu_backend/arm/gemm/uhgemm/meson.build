subdir('uhgemm_kernel')
nntrainer_inc += include_directories('uhgemm_kernel')
nntrainer_inc_abs += meson.current_source_dir() / 'uhgemm_kernel'

subdir('uhgemm_padding')
nntrainer_inc += include_directories('uhgemm_padding')
nntrainer_inc_abs += meson.current_source_dir() / 'uhgemm_padding'


uhgemm_headers = [
  'uhgemm.h',
  'uhgemm_pack.h',
  'uhgemm_common.h'
]

uhgemm_sources = [
    'uhgemm.cpp',
    'uhgemm_pack.cpp',
    'uhgemm_noTrans.cpp',
    'uhgemm_transA.cpp',
    'uhgemm_transB.cpp',
    'uhgemm_transAB.cpp',
]

foreach s : uhgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : uhgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


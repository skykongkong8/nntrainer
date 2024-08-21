uhgemm_padding_headers = [
  'uhgemm_padding.h',
  'uhgemm_padding_a.h',
  'uhgemm_padding_b.h',
]


uhgemm_padding_sources = [
  'uhgemm_padding_a.cpp',
  'uhgemm_padding_b.cpp',
]

foreach s : uhgemm_padding_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : uhgemm_padding_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach


simd_interface_sources = []
simd_interface_headers = []

arch = host_machine.cpu_family()
if arch == 'arm'
  error ('FP16/ARM code (blas_neon.cpp) uses armv8.2 instructions. armv7 is not supported.')
elif arch == 'aarch64' or get_option('platform') == 'android'
  if get_option('enable-neon')
    subdir('aarch64')
    nntrainer_inc += include_directories('aarch64')
    nntrainer_inc_abs += meson.current_source_dir() / 'aarch64'
  endif
elif get_option('enable-avx')
    subdir('x86')
    nntrainer_inc += include_directories('x86')
    nntrainer_inc_abs += meson.current_source_dir() / 'x86'
endif

foreach s : simd_interface_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : simd_interface_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

cpu_backend_headers = [
  'cpu_backend.h',
]

subdir('fallback')
nntrainer_inc += include_directories('fallback')
nntrainer_inc_abs += meson.current_source_dir() / 'fallback'

if get_option('enable-blas')
  subdir('cblas_interface')
  nntrainer_inc += include_directories('cblas_interface')
  nntrainer_inc_abs += meson.current_source_dir() / 'cblas_interface'
endif

# x86_64 (=avx)
# case 1 : cblas(fp32 avx) + fallback
# case 2 : cblas(fp32 avx) + fallback + fp16 avx2
# case 3 : full fallback 

# arm (=neon)
# case 1 : cblas(fp32 neon) + fallback + fp32 neon impl (armv7)
# case 2 : cblas(fp32 neon) + fallback + fp32 neon impl (armv8.0)
# case 3 : cblas(fp32 neon) + fp32 neon impl + fallback + fp16 neon (armv8.2-a)
# case 4 : full fallback 

arch = host_machine.cpu_family()
message('BUILD DEBUG arch ' + arch)
if arch == 'arm' or arch == 'aarch64' or get_option('platform') == 'android'
  # if get_option('enable-neon')
    subdir('arm')
    nntrainer_inc += include_directories('arm')
    nntrainer_inc_abs += meson.current_source_dir() / 'arm'
  # endif
elif arch == 'x86_64' or arch == 'x86'
  # if get_option('enable-avx')
    subdir('x86_64')
    nntrainer_inc += include_directories('x86_64')
    nntrainer_inc_abs += meson.current_source_dir() / 'x86_64'
  # endif
endif

foreach h : cpu_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

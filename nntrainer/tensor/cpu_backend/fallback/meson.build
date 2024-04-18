fallback_headers = ['fallback_internal.h']
fallback_sources = ['fallback_internal.cpp']

arch = host_machine.cpu_family()

if arch != 'arm' and arch != 'x86_64'
  fallback_headers += 'fallback.h'
  fallback_sources += 'fallback.cpp'
endif

foreach s : fallback_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : fallback_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

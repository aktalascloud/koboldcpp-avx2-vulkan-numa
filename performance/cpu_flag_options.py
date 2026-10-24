"""Named, strict-math CPU-only LTO variants. Vulkan keeps the base recipe."""
OPTIONS={
 'baseline':'',
 'tiled-haswell':'-march=haswell -mprefer-vector-width=256',
 'tiled-dot-nosched':'-fno-schedule-insns2 -fno-rename-registers',
}

def extra_flags(name):
    if name not in OPTIONS:raise RuntimeError('Unknown CPU-only compiler variant.')
    return OPTIONS[name]

def recipe(base,hot,name,force=False):
    extra=extra_flags(name)
    if name=='tiled-dot-nosched':
        base=recipe(base,hot,'tiled-haswell')
        base+='\nkcpp-quantmapper.o: CFLAGS += '+extra+'\nkcpp-quantmapper.o: CXXFLAGS += '+extra+'\n'
        if force:base+='\n.PHONY: force_backend_hot\n'+' '.join(hot)+': force_backend_hot\n'
        return base
    targets=['ggml-tiled.o','ggml-tiled-kernel.o'] if name=='tiled-haswell' else hot
    if extra:
        base+='\n'+' '.join(targets)+': CFLAGS += '+extra+'\n'+' '.join(targets)+': CXXFLAGS += '+extra+'\n'
        if name!='tiled-haswell':base+='LDFLAGS += '+extra+'\n'
    if force:
        base+='\n.PHONY: force_backend_hot\n'+' '.join(hot)+': force_backend_hot\n'
    return base

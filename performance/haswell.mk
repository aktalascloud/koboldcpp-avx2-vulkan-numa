CFLAGS += -march=haswell -mtune=haswell -mstackrealign
CXXFLAGS += -mtune=haswell -mstackrealign
FULLCFLAGS := -march=haswell -mtune=haswell
SIMPLECFLAGS := -march=haswell -mtune=haswell
SIMPLERCFLAGS := -march=haswell -mtune=haswell
FASTCFLAGS = $(CFLAGS)
FASTCXXFLAGS = $(CXXFLAGS)
expose.o: CXXFLAGS += -mno-avx -mno-avx2 -mno-fma -mno-f16c

ifeq ($(OS),Windows_NT)
    # Windows configuration.
    EXE = .exe
    RM = del /Q /F
    NVCC = nvcc
    # Windows cleanup uses backslash paths and the del command.
    CLEAN_CMD = $(RM) $(TARGET)$(EXE) output\labels\*.csv 2>NUL || exit 0
else
    # Linux configuration.
    EXE =
    RM = rm -f
    CUDA_HOME ?= /usr/local/cuda
    NVCC ?= $(shell command -v nvcc 2>/dev/null || \
                if [ -x "$(CUDA_HOME)/bin/nvcc" ]; then printf "%s" "$(CUDA_HOME)/bin/nvcc"; \
                else printf "%s" nvcc; fi)
    CLEAN_CMD = $(RM) $(TARGET) *.exe output/labels/*.csv
endif

NVFLAGS = -O3 -std=c++14
LIBS    = -lcublas

TARGET  = dpc_aknn_gpu

SRCDIR  = src
SRCS      = $(SRCDIR)/main.cu \
			$(SRCDIR)/dpc_aknn.cu \
			$(SRCDIR)/kernels.cu \
			$(SRCDIR)/utils_gpu.cu

.PHONY: all clean

all: $(TARGET)$(EXE)

$(TARGET)$(EXE): $(SRCS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

clean:
	$(CLEAN_CMD)

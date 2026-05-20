NVCC    = nvcc
NVFLAGS = -O3 -std=c++14
LIBS    = -lcublas

TARGET  = dpc_aknn_gpu
TEST    = test_dpc_aknn_gpu

SRCDIR  = src
SRCS      = $(SRCDIR)/main.cu \
			$(SRCDIR)/dpc_aknn.cu \
			$(SRCDIR)/kernels.cu \
			$(SRCDIR)/utils_gpu.cu
TEST_SRCS = $(SRCDIR)/test_dpc_aknn.cu \
			$(SRCDIR)/dpc_aknn.cu \
			$(SRCDIR)/kernels.cu \
			$(SRCDIR)/utils_gpu.cu

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)

test: $(TEST_SRCS)
	$(NVCC) $(NVFLAGS) -o $(TEST) $^ $(LIBS)
	./$(TEST)

clean:
	rm -f $(TARGET) $(TEST) *.exe output/labels/*.csv

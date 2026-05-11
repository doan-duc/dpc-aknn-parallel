CC      = gcc
CFLAGS  = -O3 -std=gnu99 -Wall -Wextra -fopenmp
LIBS    = -lm

SRCDIR  = src
TARGET  = dpc_aknn_cpu

SRCS = $(SRCDIR)/main.c \
       $(SRCDIR)/dpc_aknn_algo.c \
       $(SRCDIR)/dpc_aknn_core.c \
       $(SRCDIR)/io.c \
       $(SRCDIR)/metrics.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Ví dụ chạy với Fashion-MNIST (có ground-truth để tính metric):
run: $(TARGET)
	./$(TARGET) \
	  --input   ../../data/real/fashion-mnist/fashion_mnist_X.csv \
	  --labels  ../../data/real/fashion-mnist/fashion_mnist_y.csv \
	  --clusters 10 \
	  --k 15

clean:
	rm -f $(TARGET) output/labels/*.csv output/benchmarks/*.csv

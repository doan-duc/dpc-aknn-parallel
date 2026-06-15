ifeq ($(OS),Windows_NT)
    # Windows configuration.
    EXE = .exe
    RM = del /Q /F
    CLEAN_CMD = $(RM) $(TARGET)$(EXE) output\labels\*.csv output\benchmarks\*.csv 2>NUL || exit 0
else
    # Linux configuration.
    EXE =
    RM = rm -f
    CLEAN_CMD = $(RM) $(TARGET) *.exe output/labels/*.csv output/benchmarks/*.csv
endif

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

.PHONY: all clean

all: $(TARGET)$(EXE)

$(TARGET)$(EXE): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	$(CLEAN_CMD)

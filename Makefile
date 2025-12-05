ARCH = sm_60


HOST_COMP = mpicxx


NVCC = nvcc

NVCC_FLAGS = -arch=$(ARCH) -ccbin $(HOST_COMP) -std=c++11 -m64 -O3 -Xcompiler "-mcpu=power8 -Wall"


TARGET = solver_gpu
SRC = CUMPI.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)
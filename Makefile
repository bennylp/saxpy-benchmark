CC = cl
CC_O = /Fe:
EXE = 
CFLAGS = /Ox /EHsc /TP
RM = del

TARGETS = saxpy_cpu$(EXE) saxpy_cuda

all: $(TARGETS)

saxpy_cpu$(EXE): saxpy_cpu.cpp
		$(CC) $(CFLAGS) $(CC_O) $(@)$(EXE) saxpy_cpu.cpp

saxpy_cuda: saxpy_cuda.cu
		nvcc -Wno-deprecated-gpu-targets saxpy_cuda.cu -o saxpy_cuda
		
clean:
		$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp

#
# Instructions:
#  You need to do three steps, see below
#

###################################################################
# Step 1: Configure your compiler
###################################################################
# Windows / Visual Studio tools
CC = cl
CC_O = /Fe:
EXE = .exe
CFLAGS = /Ox /EHsc /TP
RM = del

# Linux/MacOS/Unix tools
#CC = g++
#CC_O = -o
#EXE = 
#CFLAGS = -O3 -std=c++11
#RM = rm -f


###################################################################
# Step 2: Select which implementations to enable
###################################################################
TARGET = 
TARGETS = $(TARGETS) saxpy_cpu$(EXE)
TARGETS = $(TARGETS) saxpy_cuda$(EXE)
TARGETS = $(TARGETS) saxpy_ocl1$(EXE)
TARGETS = $(TARGETS) saxpy_ocl2$(EXE)


###################################################################
# Step 3: Configure additional settings for the SDKs
###################################################################

#
# OpenCL settings
#

# Typical settings for MacOS:
#
#OCL_LDFLAGS = -framework OpenCL


# Sample settings for Windows:
#
# With OpenCL From CUDA SDK
#OCL_CFLAGS = /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"
#OCL_LDFLAGS = /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64" OpenCL.lib

# With Intel OpenCL
OCL_CFLAGS = /I"$(INTELOCLSDKROOT)/include"
OCL_LDFLAGS = /link /LIBPATH:"$(INTELOCLSDKROOT)/lib/x64" OpenCL.lib

###################################################################
# Done. 
# Usually you don't need to change anything beyond this point
###################################################################

all: $(TARGETS)

saxpy_cpu$(EXE): saxpy_cpu.cpp saxpy.h
	$(CC) $(CFLAGS) $(CC_O) $(@) saxpy_cpu.cpp

saxpy_cuda$(EXE): saxpy_cuda.cu saxpy.h
	nvcc -Wno-deprecated-gpu-targets saxpy_cuda.cu -o saxpy_cuda
		
saxpy_ocl1$(EXE): saxpy_ocl1.cpp saxpy.h
	$(CC) $(CFLAGS) $(OCL_CFLAGS) $(CC_O) $(@) saxpy_ocl1.cpp $(OCL_LDFLAGS)
	
saxpy_ocl2$(EXE): saxpy_ocl2.cpp saxpy.h
	$(CC) $(CFLAGS) $(OCL_CFLAGS) $(CC_O) $(@) saxpy_ocl2.cpp $(OCL_LDFLAGS)

clean:
	$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp *.pyc

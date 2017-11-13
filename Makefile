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
EXE = 
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
TARGETS = $(TARGETS) saxpy_cuda
#TARGETS = $(TARGETS) saxpy_ocl2$(EXE)
#TARGETS = $(TARGETS) saxpy_ocl1$(EXE)


###################################################################
# Step 3: Configure additional settings for the SDKs
###################################################################

#
# OpenCL settings
#

# Typical settings for MacOS:
#
OPENCL_LDFLAGS = -framework OpenCL
# These usually are not required on MacOS:
#OPENCL_DIR = /System/Library/Frameworks/OpenCL.framework
#OPENCL_DIR = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk/System/Library/Frameworks/OpenCL.framework
#OPENCL_CFLAGS = -I$(OPENCL_DIR)/Versions/A/Headers

# Sample settings for Windows:
#


###################################################################
# Done. 
# Usually you don't need to change anything beyond this point
###################################################################

all: $(TARGETS)

saxpy_cpu$(EXE): saxpy_cpu.cpp saxpy.h
	$(CC) $(CFLAGS) $(CC_O) $(@)$(EXE) saxpy_cpu.cpp

saxpy_cuda: saxpy_cuda.cu saxpy.h
	nvcc -Wno-deprecated-gpu-targets saxpy_cuda.cu -o saxpy_cuda
		
saxpy_ocl2$(EXE): saxpy_ocl2.cpp saxpy.h
	$(CC) $(CFLAGS) $(OPENCL_CFLAGS) $(OPENCL_LDFLAGS) $(CC_O) $(@)$(EXE) saxpy_ocl2.cpp

saxpy_ocl1$(EXE): saxpy_ocl1.cpp saxpy.h
	$(CC) $(CFLAGS) $(OPENCL_CFLAGS) $(OPENCL_LDFLAGS) $(CC_O) $(@)$(EXE) saxpy_ocl1.cpp
	
clean:
	$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp *.pyc

# Windows / Visual Studio tools
#CC = cl
#CC_O = /Fe:
#EXE = 
#CFLAGS = /Ox /EHsc /TP
#RM = del

# Linux/MacOS/Unix tools
CC = g++
CC_O = -o
EXE = 
CFLAGS = -O3 -std=c++11
RM = rm -f


# Choose targets:
TARGET = 
TARGETS := $(TARGETS) cpu$(EXE)
#TARGETS := $(TARGETS) cuda
TARGETS := $(TARGETS) opencl2$(EXE)
TARGETS := $(TARGETS) opencl1$(EXE)

# OpenCL settings:
#OPENCL_DIR = /System/Library/Frameworks/OpenCL.framework
#OPENCL_DIR = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk/System/Library/Frameworks/OpenCL.framework
#OPENCL_CFLAGS = -I$(OPENCL_DIR)/Versions/A/Headers
OPENCL_LDFLAGS = -framework OpenCL


all: $(TARGETS)

cpu$(EXE): cpu.cpp saxpy.h
	$(CC) $(CFLAGS) $(CC_O) $(@)$(EXE) cpu.cpp

cuda: cuda.cu saxpy.h
	nvcc -Wno-deprecated-gpu-targets cuda.cu -o cuda
		
opencl2$(EXE): opencl2.cpp saxpy.h
	$(CC) $(CFLAGS) $(OPENCL_CFLAGS) $(OPENCL_LDFLAGS) $(CC_O) $(@)$(EXE) opencl2.cpp

opencl1$(EXE): opencl1.cpp saxpy.h
	$(CC) $(CFLAGS) $(OPENCL_CFLAGS) $(OPENCL_LDFLAGS) $(CC_O) $(@)$(EXE) opencl1.cpp
	
clean:
	$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp

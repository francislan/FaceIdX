IDIR = ../include
CC = /usr/local/cuda/bin/nvcc
CFLAGS = -I$(IDIR) -arch=sm_30 -fmad=false

ODIR = obj

_DEPS = misc.h stb_image.h eigen_cpu.h eigen_gpu.h database_cpu.h database_gpu.h stb_image_write.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = FaceIdX.o eigen_cpu.o eigen_gpu.o database_cpu.o database_gpu.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

FaceIdX: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~

IDIR = ../include
CC = /usr/local/cuda/bin/nvcc
CFLAGS = -I$(IDIR) -arch=sm_30 -fmad=false

ODIR = obj

_DEPS = nice_print.h stb_image.h eigen.h database.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = FaceIdX.o eigen.o database.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

FaceIdX: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~

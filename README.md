# FaceIdX
GPU aXelerated Face Identification

To make:
Go to /code/src directory,
mkdir obj
make

To save eigenfaces and reconstructed faces to disk, you have to create the
'eigen' and 'reconstructed' directories beforehand (in /code/src)

To run the GPU version:
./FaceIdX

To run the CPU version:
./FaceIdX -cpu

Using 2 single-file public domain librairies for image reading/writing:
https://github.com/nothings/stb.git by Sean T. Barrett

The faces in the dataset are from:
- The Yale dataset of faces
- Internet (for celebraties)
- Our pictures (for us)

Francis Lan and XiangLu Kong
Project for W4995 - GPU Computing, Spring 2015

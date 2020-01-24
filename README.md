# Topology Classification
Codebase to reproduce dataset for https://arxiv.org/abs/1807.00083

1. To create the raw image:
The raw image is set to png because of the irregularity of detector resolution.
To create an image from the input h5 file, do:
```
python TransformToRawImage.py [inputFile] [lowerLimit] [upperLimit]
```
where the arguments are all optional. If no input file is specified, it will transform the whole input directory specified in the source code. 

`lowerLimit` and `upperLimit` indicate the number of events to transform within the input file.

Before running the code, please modify the input and output directory in the source code.

# Topology Classification
Codebase to reproduce dataset for https://arxiv.org/abs/1807.00083

## To create the raw image
The raw image is set to png because of the irregularity of detector resolution.
To create an image from the input h5 file, do:
```
python TransformToRawImage.py [inputFile] [lowerLimit] [upperLimit]
```
where the arguments are all optional. If no input file is specified, it will transform the whole input directory specified in the source code. 

`lowerLimit` and `upperLimit` indicate the number of events to transform within the input file.

Before running the code, please modify the input and output directory in the source code.

We also have the version of using numpy arrays instead of PNG images, but we observed either similar or worse performances so we did not include it in the paper:

* `TransformArray.py`: pile up each component of the detector as a separated channel, resulting in a 3D matrix.
* `TransformOneArray.py`: append each component next to each other as a flat and long 2D matrix, similar to the PNG image.

The usage is the same as `TransformToRawImage.py`

## To creat the abstract image

Same usage as above:
```
python TransformToAbstractImage.py [inputFile] [lowerLimit] [upperLimit]
```
and remember to change the input/output directory


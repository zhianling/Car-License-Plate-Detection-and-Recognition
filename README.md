# Car Plate Detection & OCR

## Development
Create and develop in your own branch:

an - Zhi An

sia - Sia

alfi - Alfi

natalie - Natalie

joanne - Joanne

Once the branch is ready to merge to the main branch, create a pull request.

## Clone This Repository
```bash
git clone https://github.com/zhianling/Car-License-Plate-Detection-and-Recognition.git
```

## Get The Raw Dataset
Download the raw dataset from https://drive.google.com/file/d/1v0gIs3jtA1krnvqMwOgswKludjOggh-E/view?usp=sharing.

## Get The Models
Download the models from https://drive.google.com/file/d/1QfOAmHzjnUlpJQX-oAFXz42BOgV7NpAB/view?usp=sharing.

## Conda environment
To construct conda environment from environment.yml:
```bash
conda env create -f environment.yml
```

## Code Structure
`Pipeline` directory - Test pipeline of object detection and OCR models

`Program` directory - Working demo of the program. In the directory, download the models.7z and unzip. On terminal 
run `python ./car_plate_gui.py`.

`Train` directory - Contains training scripts for YOLOv8, Faster R-CNN, and EasyOCR.
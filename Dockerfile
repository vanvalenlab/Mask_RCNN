# Use the nvidia tensorflow:18.04-py3 image as the parent image

FROM nvcr.io/vvlab/tensorflow:18.04-py3



# System maintenance

RUN apt update && apt-get install -y python3-tk

RUN pip install --upgrade pip



# Set working directory

WORKDIR /Mask_RCNN



# Copy the deepcell-tf requirements.txt and install its dependencies

COPY ./requirements.txt /lib/Mask_RCNN/requirements.txt

RUN pip install -r /lib/Mask_RCNN/requirements.txt

RUN pip install opencv-python

RUN apt update && apt install -y libsm6 libxext6

RUN apt-get install -y libsm6

RUN apt-get install -y libxrender-dev



COPY . /Mask_RCNN



# Make port 80 available to the world outside this container

EXPOSE 80

# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile
FROM python:3.8-slim

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /code

RUN apt-get update -y
#RUN apt-get install -y python3 python3-pip
RUN apt install git --fix-missing -y
RUN apt install wget -y

# installing other libraries
RUN apt-get install python3-pip -y && \
    apt-get -y install sudo
RUN apt-get install curl -y
RUN apt-get install nano -y
RUN apt-get update && apt-get install -y git
RUN apt-get install libblas-dev -y && apt-get install liblapack-dev -y
RUN apt-get install gfortran -y
RUN apt-get install libpng-dev -y
RUN apt-get install python3-dev -y

WORKDIR /code

# install dependencies
COPY ./demo/requirements.txt /code/demo/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/demo/requirements.txt

# resolve issue with tf==2.4 and gradio dependency collision issue
RUN pip install --force-reinstall typing_extensions==4.7.1

# lower pydantic version to work with typing_extensions deprecation
RUN pip install --force-reinstall "pydantic<2.0.0"

# Install wget
RUN apt install wget -y && \
    apt install unzip

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# download test data
RUN wget "https://github.com/jpdefrutos/DDMR/releases/download/test_data_brain_v0/ixi_image.nii.gz" && \
    wget "https://github.com/jpdefrutos/DDMR/releases/download/test_data_brain_v0/ixi_image2.nii.gz"

# CMD ["/bin/bash"]
CMD ["python3", "app.py"]
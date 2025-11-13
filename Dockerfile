FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
LABEL hostname="syntra-docker"

# Set the working directory
WORKDIR /project

# Copy the setup script and requirements file
COPY requirements.txt /project/requirements.txt

# Update conda and install system dependencies
RUN conda update conda -y
RUN apt-get update
RUN apt install python3-dev  -y
RUN pip install --upgrade pip 

# Install faiss (from listed conda channels)
RUN conda install -y -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=11.4,<=11.8'
# Install dependencies for training
RUN pip install xformers==0.0.29
RUN pip install -r requirements.txt

WORKDIR /workspace
# Start with nvidia-enabled pytorch image
FROM nvcr.io/nvidia/pytorch:23.01-py3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Set path environment to include the code
ENV PATH=$PATH:/workspace/code/
ENV PYTHONPATH=${PYTHONPATH}:/workspace/code/

# Expose port 8888 for Jupyter notebook server
EXPOSE 8888

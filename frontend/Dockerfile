# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYHTONUNBUFFERED=1
RUN apt-get update
RUN apt-get -y install python-dev-is-python3 libatlas-base-dev build-essential cmake --fix-missing
RUN apt-get -y install ffmpeg libsm6 libxext6 --fix-missing
RUN apt-get -y install python3-numpy --fix-missing

# Set the working directory in the container
WORKDIR /code

# Make port 3000 available to the world outside this container
EXPOSE 3000


# Install any needed packages specified in requirements.txt
COPY ./heavy_requirements.txt /code/heavy_requirements.txt
RUN pip install --upgrade -r /code/heavy_requirements.txt

RUN pip install cmake

# Install any needed packages specified in requirements.txt
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade -r /code/requirements.txt

RUN pip install imutils
RUN pip install python-multipart

# Copy the current directory contents into the container at /code
COPY ./ /code/

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]

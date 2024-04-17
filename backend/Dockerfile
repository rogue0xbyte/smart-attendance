# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYHTONUNBUFFERED=1
RUN apt-get update
RUN apt-get -y install python-dev-is-python3 libatlas-base-dev build-essential cmake --fix-missing
RUN apt-get -y install ffmpeg libsm6 libxext6 --fix-missing
RUN apt-get -y install python3-numpy --fix-missing

# Set the working directory in the container
WORKDIR /code

# Make port 8000 available to the world outside this container
EXPOSE 8000

RUN pip install dlib==19.17.0
RUN pip install opencv-python==4.9.0.80

# Install any needed packages specified in requirements.txt
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade -r /code/requirements.txt


# Install any needed packages specified in new_requirements.txt
COPY ./update_requirements.txt /code/update_requirements.txt
RUN pip install --upgrade -r /code/update_requirements.txt

# Copy the current directory contents into the container at /code
COPY ./ /code/

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload"]

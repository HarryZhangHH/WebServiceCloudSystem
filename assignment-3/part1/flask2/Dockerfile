FROM python:3.8

# set a directory for the app
WORKDIR /url

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# define the port number the container should expose
EXPOSE 5003

# run the command
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=5003"]
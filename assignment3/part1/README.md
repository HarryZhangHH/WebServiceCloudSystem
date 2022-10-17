# wscs_assignment3.1 - Container Virtualization

## File Introduction
- docker-compose.yml is used to build and run containers used for this assignment
- flask1 contains the file needed to build and run for login service
- flask2 contains the file needed to build and run for url-shortener service
- nginx contains the file needed to build and run for nginx reverse proxy service(to implement bonus)

## Run(under this folder)
```
sudo docker-compose build
sudo docker-compose up -d
```
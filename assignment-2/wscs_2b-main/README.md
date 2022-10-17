# wscs_assignment2 - RESTful microservices architectures bonus point implementation

## Code Introduction
- main.py is used to connect front and back
- init_db.py is used to initilize the database
- schema.sql contains the structure of the url database
- models.py contains the structure of the user database
- templates/index.html html page for the shortner function
- templates/stats.html html page for data exhibition
- templates/edit.html html page for edit the data
- templates/login.html html page for login
- templates/signup.html html page for signup

## Prerequisites

```
python==3.8(recommend creating virtual environment using conda)
flask==2.0.3
flask-jwt-extended==4.3.1
flask-sqlalchemy==2.5.1
werkzeug==2.0.3
hashids==1.3.1
sqlite==3.38.2
```

## Run
```
flask run (open 127.0.0.1:5000 in browser)
```
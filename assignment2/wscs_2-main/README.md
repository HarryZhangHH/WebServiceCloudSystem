# wscs_assignment2 - RESTful microservices architectures

## Code Introduction
- main.py is used to perform user login function
- init_db.py is used to initilize the database
- schema.sql contains the structure of the url database
- models.py contains the structure of the user database
- app.py contains the URL-Shortener service, but mostly require jwt token

## Prerequisites

```
python==3.8(recommend creating virtual environment using conda)
flask==2.0.3
flask-jwt-extended==4.3.1
flask-sqlalchemy==2.5.1
werkzeug==2.0.3
hashids==1.3.1
sqlite==3.38.2
validators==0.18.2
```

## Run
```
export FLASK_RUN_PORT=4000
export FLASK_APP=main
flask run (use 127.0.0.1:4000)
export FLASK_RUN_PORT=5000
export FLASK_APP=app
flask run (use 127.0.0.1:5000)
```
## Reference
https://flask-jwt-extended.readthedocs.io/en/stable/  
https://www.adamsmith.haus/python/answers/how-to-return-http-status-code-201-using-flask-in-python  
https://www.digitalocean.com/community/tutorials/how-to-add-authentication-to-your-app-with-flask-login  

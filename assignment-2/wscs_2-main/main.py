from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, make_response
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity, set_access_cookies, unset_jwt_cookies
)
from werkzeug.security import generate_password_hash, check_password_hash
from models import User
from db import db
import sqlite3
from hashids import Hashids

def create_app():
    app = Flask(__name__)
    # SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    # set jwt stored to cookies
    db.init_app(app)
    app.config["JWT_SECRET_KEY"] = "super-secret"  # Change this "super secret" with something else!
    return app

app = create_app()
# Initialize the jwt
jwt = JWTManager(app)
hashids = Hashids(min_length=4, salt=app.config['SECRET_KEY'])

# the register page
@app.route('/users', methods=["POST"])
def register():
    # get the username and password
    username = request.args.get("Username")
    password = request.args.get("password")
    user = User.query.filter_by(username=username).first()
    if user:
        return jsonify({"msg": "User already exists"}), 401
    # insert into database
    new_user = User(username=username, password=generate_password_hash(password, method='sha256'))

    db.session.add(new_user)
    db.session.commit()
    return jsonify(success=True)

# the login page for user
@app.route('/users/login', methods=["POST"])
def login():
    # get the username and password
    username = request.args.get("Username")
    password = request.args.get("password")

    # check whether the user is in the database
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return "forbidden", 403
    # generate access token for user
    access_token = create_access_token(identity=user.id)
    # return the access token
    return access_token, 200

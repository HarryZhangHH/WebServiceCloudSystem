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

# Connect to the database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_app():
    app = Flask(__name__)
    # SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    # set jwt stored to cookies
    app.config['JWT_TOKEN_LOCATION'] = ['cookies']
    db.init_app(app)
    app.config["JWT_COOKIE_SECURE"] = False
    app.config["JWT_SECRET_KEY"] = "super-secret"  # Change this "super secret" with something else!
    return app

app = create_app()
# Initialize the jwt
jwt = JWTManager(app)
hashids = Hashids(min_length=4, salt=app.config['SECRET_KEY'])

# if not login then redirect to login page
@jwt.unauthorized_loader
def custom_unauthorized_response(_err):
    return redirect(url_for('login'))

# the main page, user needs to login to access it
@app.route('/', methods=["GET"])
@jwt_required()
def index():
    # if the get request then go to main page
    return render_template('index.html')

@app.route('/', methods=["POST"])
def shorten():
# get the url from the html
    conn = get_db_connection()
    url = request.form['url']

    # if url is empty
    if not url:
        flash('The URL is required!')
        return redirect(url_for('index'))

    # if url doesn't include https
    if "https://" not in url:
        url = "https://" + url

    # insert into database
    url_data = conn.execute('INSERT INTO urls (original_url) VALUES (?)',
                                (url,))
    conn.commit()
    conn.close()

    # generate shortened id
    url_id = url_data.lastrowid
    hashid = hashids.encode(url_id)
    short_url = request.host_url + hashid

    # reflected on the html
    return render_template('index.html', short_url=short_url)

# when trying to visit the shortened html
@app.route('/<id>')
def url_redirect(id):
    conn = get_db_connection()

    # get the original_id
    original_id = hashids.decode(id)
    if original_id:
        original_id = original_id[0]
        # query from the database
        url_data = conn.execute('SELECT original_url, clicks FROM urls'
                                ' WHERE id = (?)', (original_id,)
                                ).fetchone()
        original_url = url_data['original_url']
        clicks = url_data['clicks']

        # update the clicks number: +1
        conn.execute('UPDATE urls SET clicks = ? WHERE id = ?',
                     (clicks+1, original_id))

        conn.commit()
        conn.close()
        # redirect for original web page
        return redirect(original_url)
    else:
        # if not in the database, then redirect to the main page
        flash('Invalid URL')
        return redirect(url_for('index'))

# stats page are used to display the database information, also need to login to access
@app.route('/stats')
@jwt_required()
def stats():
    conn = get_db_connection()
    # get all records from the database
    db_urls = conn.execute('SELECT id, created, original_url, clicks FROM urls'
                           ).fetchall()
    conn.close()

    # get the shortened url for each record
    urls = []
    for url in db_urls:
        url = dict(url)
        url['short_url'] = request.host_url + hashids.encode(url['id'])
        urls.append(url)

    # reflect in the html page
    return render_template('stats.html', urls=urls)

# edit the record, to edit it, user need to login
@app.route('/stats/<int:id>/edit/', methods=["GET"])
@jwt_required()
def edit(id):
    return render_template('edit.html')

@app.route('/stats/<int:id>/edit/', methods=["POST"])
def change(id):
    conn = get_db_connection()
    # in the edit page
    # get the url from the text line
    new_url = request.form['url']
    # check whether the new url satisfies the need
    if not new_url:
        flash('New url is required!')
        return redirect(url_for('edit', id=id))
    if "https://" not in new_url:
        new_url = "https://" + new_url
        # update the database
    conn.execute('UPDATE urls SET original_url = ? WHERE id = ?',
                     (new_url, id))
    conn.commit()
    conn.close()
    # redirect to the stats page
    return redirect(url_for('stats'))

# if want to delete a record from the database
@app.route('/stats/<int:id>/delete/', methods=('POST',))
def delete(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM urls WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('stats'))

# if want to clear all records from the database
@app.route('/stats/clear/', methods=('POST',))
def clear():
    conn = get_db_connection()
    conn.execute('DELETE FROM urls')
    conn.commit()
    conn.close()
    return redirect(url_for('stats'))

# this function is used to check the name of the name
@app.route('/profile')
@jwt_required()
def profile():
    # Access the identity of the current user with get_jwt_identity
    current_user_id = get_jwt_identity()
    user = User.query.get(int(current_user_id))
    return render_template('profile.html', name=user.name)

# This function is used to login
@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
         # login code goes here
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        # check if the user actually exists
        # take the user-supplied password, hash it, and compare it to the hashed password in the database
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login')) # if the user doesn't exist or password is wrong, reload the page

        # generate the jwt
        access_token = create_access_token(identity=user.id)
        response = make_response(redirect(url_for('profile')))
        # store it in the cookies
        set_access_cookies(response,access_token)
        return response
    return render_template('login.html')

# This function is used to register user
@app.route('/signup', methods=('GET', 'POST'))
def signup():
    if request.method == 'POST':
        # code to validate and add user to database goes here
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

        if user: # if a user is found, we want to redirect back to signup page so user can try again
            flash('Email address already exists')
            return redirect(url_for('signup'))

        # create a new user with the form data. Hash the password so the plaintext version isn't saved.
        new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

        # add the new user to the database
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

# This function is used to logout
@app.route('/logout')
@jwt_required()
def logout():
    response = make_response(redirect(url_for('login')))
    # remove jwt from the cookies
    unset_jwt_cookies(response)
    return response

from flask import Flask, request, jsonify, make_response
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity)
import sqlite3
from hashids import Hashids
import validators

# Connect to the database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# the secret key for hash table
app = Flask(__name__)
app.config['SECRET_KEY'] = 'this should be a secret random string'
app.config["JWT_SECRET_KEY"] = "super-secret"  # Change this "super secret" with something else!
jwt = JWTManager(app)
hashids = Hashids(min_length=4, salt=app.config['SECRET_KEY'])

# if not login then redirect to login page
@jwt.unauthorized_loader
def custom_unauthorized_response(_err):
    return "forbidden", 403

# the main page
@app.route('/', methods=('GET', 'POST', 'DELETE'))
@jwt_required()
def index():
    conn = get_db_connection()

    # if use the post method
    if request.method == 'POST':
        # get the url from the header
        url = request.args.get("url")

        # if url doesn't include https
        if "https://" not in url:
            url = "https://" + url
        
        # check the correctness of the url
        if validators.url(url)!=True:
            return "error", 400

        # get the current user id
        current_user_id = get_jwt_identity()
        # insert into database with the information of the url and user id
        url_data = conn.execute('INSERT INTO urls (original_url, user_id) VALUES (?, ?)', (url, current_user_id))
        conn.commit()
        conn.close()

        # generate shortened id
        url_id = url_data.lastrowid
        hashid = hashids.encode(url_id)

        # return the shortened url
        return hashid, 201
    # no all delete function
    elif request.method == 'DELETE':
        conn.close()
        return "not accessible", 404
    # if use get function
    # get current user first
    current_user_id = get_jwt_identity()
    # then query for the records this user create
    db_urls = conn.execute('SELECT id, created, original_url, user_id, clicks FROM urls WHERE user_id = (?)', (current_user_id,)).fetchall()
    conn.close()
    # get the shortened url for each record
    urls = []
    for url in db_urls:
        url = dict(url)
        url['short_url'] = request.host_url + hashids.encode(url['id'])
        urls.append(url)
    # return the information
    return jsonify(urls), 200

@app.route('/<id>', methods=['GET'])
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
        if url_data:
            original_url = url_data['original_url']
            clicks = url_data['clicks']
            # update the clicks number: +1
            conn.execute('UPDATE urls SET clicks = ? WHERE id = ?',
                         (clicks+1, original_id))

            conn.commit()
            conn.close()
            # return the original url
            return original_url, 301
        else:
            conn.close()
            return "not found", 404
    else:
        conn.close()
        return "not found", 404

@app.route('/<id>', methods=('PUT', 'DELETE'))
@jwt_required()
def edit(id):
    if request.method == 'PUT':
        conn = get_db_connection()
        # get the id of the url from the database
        original_id = hashids.decode(id)
        if original_id:
            original_id = original_id[0]
            # get the user id
            current_user_id = get_jwt_identity()
            # check whether the record from the database is created by the current user
            url_data = conn.execute('SELECT original_url, clicks FROM urls'
                ' WHERE id = (?) AND user_id = (?)', (original_id, current_user_id)).fetchone()
            if url_data:
                # if so, get new url
                new_url = request.args.get("new_url")
                if "https://" not in new_url:
                    new_url = "https://" + new_url
                # check the correctness of the new url
                if validators.url(new_url)!=True:
                    return "error", 400
                # update the database
                conn.execute('UPDATE urls SET original_url = ? WHERE id = ?',
                     (new_url, original_id))
                conn.commit()
                conn.close()
                return "update successfully", 200
            else:
                conn.close()
                return "not found", 404
        else:
            conn.close()
            return "not found", 404
        
    else:
        # if delete the record
        conn = get_db_connection()
        original_id = hashids.decode(id)
        if original_id:
            # get the original id
            original_id = original_id[0]
            # get the current user 
            current_user_id = get_jwt_identity()
            # check whether the url is created by the user
            url_data = conn.execute('SELECT original_url, clicks FROM urls'
                ' WHERE id = (?) AND user_id = (?)', (original_id, current_user_id)).fetchone()
            if url_data:
                # if so, delete the record
                conn.execute('DELETE FROM urls WHERE id = ?', (original_id,))
                conn.commit()
                conn.close()
                return "delete successfully", 204
            else:
                conn.close()
                return "not found", 404
        else:
            conn.close()
            return "not found", 404

if __name__ == "__main__":
    app.run(debug=True, port=5003)

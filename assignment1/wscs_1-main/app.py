import sqlite3
from hashids import Hashids
from flask import Flask, render_template, request, flash, redirect, url_for

# Connect to the database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# the secret key for hash table
app = Flask(__name__)
app.config['SECRET_KEY'] = 'this should be a secret random string'

hashids = Hashids(min_length=4, salt=app.config['SECRET_KEY'])

# the main page
@app.route('/', methods=('GET', 'POST'))
def index():
    conn = get_db_connection()

    # if press the submit button
    if request.method == 'POST':
        # get the url from the html
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

    # if the get request then go to main page
    return render_template('index.html')

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

# stats page are used to display the database information
@app.route('/stats')
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

# edit the record
@app.route('/stats/<int:id>/edit/', methods=('POST','GET'))
def edit(id):
    conn = get_db_connection()
    # in the edit page
    if request.method == 'POST':
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
    # if get then go to the edit page
    return render_template('edit.html')

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
    

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import requests

# Load environment variables (like your API key)
GEOCODING_API_KEY = ('595604d27a984913acdce947d8a70f54')

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# Connect to SQLite database
def get_db():
    return sqlite3.connect('geocache.db')

# Geocode endpoint: Checks cache first, then calls OpenCage API
@app.route('/geocode')
def geocode():
    city = request.args.get('city', '')
    country = request.args.get('country', '')
    
    if not city:
        return jsonify({'error': 'City is required!'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Check if city+country is already cached
    cached = cursor.execute('''
        SELECT lat, lon FROM cities 
        WHERE city = ? AND country = ?
    ''', (city, country)).fetchone()

    if cached:
        conn.close()
        return jsonify({'lat': cached[0], 'lon': cached[1]})

    # If not cached, call OpenCage API
    try:
        response = requests.get(
            f'https://api.opencagedata.com/geocode/v1/json?q={city},{country}&key={os.getenv("OPENCAGE_API_KEY")}'
        )
        data = response.json()
        if data['results']:
            lat = data['results'][0]['geometry']['lat']
            lon = data['results'][0]['geometry']['lng']

            # Save to database
            cursor.execute('''
                INSERT INTO cities (city, country, lat, lon) 
                VALUES (?, ?, ?, ?)
            ''', (city, country, lat, lon))
            conn.commit()
            conn.close()

            return jsonify({'lat': lat, 'lon': lon})
        else:
            return jsonify({'error': 'City not found!'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

# Define the directory where text files are stored
input_dir = ('/home/tabidah/kafka/project/amna-project/songs_files_by_album')
audio_dir = ('/home/tabidah/kafka/project/sample_audio')
csv_data = pd.read_csv("/home/tabidah/kafka/project/merged_data.csv")


# Define a function to get the list of artists from text files
def get_artists_from_files():
    artists = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            # Extract artist name from file name
            artist_name = file_name.replace('.txt', '').replace('_', ' ')
            artists.append(artist_name)
    return artists

def parse_recommendations(file_path):
    recommendations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        track_info = {}
        for line in lines:
            if line.strip():  # Check if line is not empty
                key_value = line.split(': ')
                if len(key_value) == 2:
                    key, value = key_value
                    track_info[key.strip()] = value.strip()
                else:
                    # Append the previous track_info to recommendations
                    if track_info:
                        recommendations.append(track_info)
                    # Start a new track_info
                    track_info = {}
        # Append the last track_info
        if track_info:
            recommendations.append(track_info)
    return recommendations

def get_recommendations_from_file(artist_name):
    file_name = artist_name + '.txt'  # Construct the file name
    file_path = os.path.join(input_dir, file_name)  # Construct the file path
    recommendations = parse_recommendations(file_path)  # Parse recommendations from the file
    return recommendations

@app.route('/')
def home():
    """
    Home page route.
    Displays a list of artists.
    """
    # Get the list of artists from text files
    artists = get_artists_from_files()
    return render_template('home_page.html', artists=artists)

@app.route('/recommendations/<artist_name>')
def recommendations(artist_name):
    """
    Recommendations page route.
    Displays recommendations for the selected artist.
    """

    recommendations_data = get_recommendations_from_file(artist_name)
    if recommendations_data:
        return render_template('recommendations.html', artist_name=artist_name, recommendations_data=recommendations_data)
    else:
        return f"No recommendations available for {artist_name}."
        
from flask import send_file

@app.route('/streaming/<track_id>')
def streaming(track_id):
    """
    Streaming page route.
    Streams the audio file corresponding to the given track ID.
    """
    formatted_track_id = track_id.zfill(6)
    audio_file_path = f"/home/tabidah/kafka/project/sample_audio/{formatted_track_id}.mp3"
    return send_file(audio_file_path, mimetype='audio/mpeg')
    
@app.route('/audio/<path:filename>')
def download_file(filename):
    return send_from_directory('/home/tabidah/kafka/project/sample_audio', filename)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, Response
import cv2
import pandas as pd
from deepface import DeepFace
import tensorflow as tf
from deepface.detectors import FaceDetector
import numpy as np
from statistics import mode
import socket
import spotipy
import pickle
import base64
import requests
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from youtubesearchpython import *

global mood_songs
global current_song_id
global duration, pred_string
global song_link


pred_string = ""
song_link = "yWHrYNP6j4k"

app = Flask(__name__)

model = tf.keras.models.load_model('saved_model/model_t2')
mood_songs = pd.read_csv("Ansh_moods.csv")

model_moods = pickle.load(open("moods.pkl", "rb"))

CLIENT_ID = "29552cf19dd74bdc9ca2bf1af6a555e4"
CLIENT_SECRET = "7e8cac1fa75f480cad828a7c391044a5"
SCOPE = 'user-read-playback-state user-modify-playback-state'

client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
client_credentials_base64 = base64.b64encode(client_credentials.encode())

token_url = 'https://accounts.spotify.com/api/token'
headers = {
    'Authorization': f'Basic {client_credentials_base64.decode()}',
    "Scope": SCOPE
}
data = {
    'grant_type': 'client_credentials'
}
response = requests.post(token_url, data=data, headers=headers)

if response.status_code == 200:
    access_token = response.json()['access_token']
    print("Access token obtained successfully.")
else:
    print("Error obtaining access token.")
    exit()

current_song_id = None
duration = None

all_tracks = pd.read_csv("data.csv")
all_tracks.drop("year", axis = 1, inplace = True)
all_tracks = all_tracks.sort_values(by='popularity', ascending=False).head(10000)

def content_based_recommendations(music_df, num_recommendations=20):
    scaler = MinMaxScaler()
    music_features = music_df[['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness',
       'liveness', 'valence', 'loudness', 'speechiness', 'tempo', 'key']].values
    music_features_scaled = scaler.fit_transform(music_features)
    all_tracks_features_scaled = scaler.fit_transform(all_tracks[['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness',
       'liveness', 'valence', 'loudness', 'speechiness', 'tempo', 'key']])

    similarity_scores = cosine_similarity(music_features_scaled, all_tracks_features_scaled)

    similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations * 15 + 1]

    artists = [item for sublist in music_df["artists"] for item in sublist]

    content_based_rec = all_tracks.iloc[similar_song_indices][['name', 'artists', 'popularity', 'id']]
    content_based_rec["id"] = "https://open.spotify.com/track/" + content_based_rec["id"].str[:]

    for artist in set(content_based_rec["artists"]).intersection(set(artists)):
        content_based_rec.loc[content_based_rec["artists"] == artist, 'popularity'] += 10
    content_based_rec = content_based_rec.sort_values(by="popularity", ascending = False)
    content_based_rec = content_based_rec.loc[~content_based_rec['name'].isin(music_df['name'])].copy()

    return content_based_rec.head(num_recommendations)

def get_trending_playlist_data(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)

    music_data = []
    i = 0
    while len(sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))', offset = i * 100)['items']):
        playlist_tracks = sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))', offset = i * 100)
        for track_info in tqdm.tqdm(playlist_tracks['items']):
            try:
                track = track_info['track']
                track_name = track['name']
            except:
                continue

            artists = ', '.join([artist['name'] for artist in track['artists']])
            album_id = track['album']['id']
            track_id = track['id']

            audio_features = sp.audio_features(track_id)[0] if track_id != 'Not available' else None

            try:
                album_info = sp.album(album_id) if album_id != 'Not available' else None
                release_date = album_info['release_date'] if album_info else None
            except:
                release_date = None

            try:
                track_info = sp.track(track_id) if track_id != 'Not available' else None
                popularity = track_info['popularity'] if track_info else None
            except:
                popularity = None

            track_data = {
                'name': track_name,
                'artists': artists,
                'popularity': popularity,
                'release_date': release_date,
                'id' : track_id if track_id != "Not available" else None,
                'duration_ms': audio_features['duration_ms'] if audio_features else None,
                'explicit': track_info.get('explicit', None),
                'danceability': audio_features['danceability'] if audio_features else None,
                'energy': audio_features['energy'] if audio_features else None,
                'key': audio_features['key'] if audio_features else None,
                'loudness': audio_features['loudness'] if audio_features else None,
                'mode': audio_features['mode'] if audio_features else None,
                'speechiness': audio_features['speechiness'] if audio_features else None,
                'acousticness': audio_features['acousticness'] if audio_features else None,
                'instrumentalness': audio_features['instrumentalness'] if audio_features else None,
                'liveness': audio_features['liveness'] if audio_features else None,
                'valence': audio_features['valence'] if audio_features else None,
                'tempo': audio_features['tempo'] if audio_features else None,
            }

            music_data.append(track_data)

        i += 1

    df = pd.DataFrame(music_data)

    features = df[['popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness',
       'liveness', 'valence', 'loudness', 'speechiness', 'tempo', 'key']]

    predictions = model_moods.predict(features)

    df["mood"] = predictions
    # df.to_csv("Ansh_moods.csv", index = True)

    similar_df = content_based_recommendations(df)

    return pd.concat([similar_df, df])

def get_class_arg(array):
    string = ""
    classes = ["Neutral", "Disgust", "Sad", "Surprise", "Angry", "Happy", "Neutral"]
    for i in [0, 2, 4, 5]:
        string += f"{classes[i]} : {round(array[0][i] * 100, 2)}   "
    return string

def get_class(argument):
    return ["Fear", "Disgust", "Sad", "Surprise", "Angry", "Happy", "Neutral"][argument]

def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    prev_mood = []
    mode_mood = "Happy"

    emotions = [[], [], [], [], [], [], []]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if 1:
            cv2.imwrite("Face.jpg", frame)
            try:
                obj = DeepFace.analyze(img_path="Face.jpg", actions=["emotion"], silent = True)
                detector = FaceDetector.build_model('opencv')
                faces_1 = FaceDetector.detect_faces(detector, 'opencv', frame)
                dim = faces_1[0][1]
                cv2.rectangle(frame, (dim[0], dim[1]), (dim[0] + dim[2], dim[1] + dim[3]), (140, 140, 0), 2)
                roi = frame[dim[1]:dim[1] + dim[3], dim[0]:dim[0] + dim[2]]
                img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(roi, (48, 48))
                img = img / 255.0
                pred = model.predict(np.array([img]))
                pred[0][1] = 0
                pred[0][6] = 0
                # pred[0][4] = 0
                pred[0][5] *= 60
                pred[0][2] *= 1.4
                pred_sum = sum(pred[0])
                for i in range(7):
                    pred[0][i] = pred[0][i] / pred_sum
                for i in range(0, 7):
                    emotions[i].append(pred[0][i])

                pred_string = get_class_arg(pred)

                max_pred = max(pred[0])
                j = 0
                mood = "Happy"
                for i in [0, 2, 4, 5]:
                    if max_pred == pred[0][i]:
                        mood = ["Happy", "Sad", "Sad", "Energetic"][j]
                    else:
                        j += 1

                prev_mood.append(mood)
                if(mode(prev_mood[-10:]) != mode_mood):
                    print("ANSH")
                    mode_mood = mode(prev_mood[-10:])
                    current_mood = mood_songs.loc[mood_songs["mood"] == mood]

                    global current_song_id, duration, song_link
                    song = current_mood.sample(n=1)
                    current_song_id = song["id"].values[0]
                    duration = song["duration_ms"].values[0]
                    song_name = song['name'].values[0] + " " + song['artists'].values[0]

                    customSearch = VideosSearch(song_name + " Lyrics", limit = 1)

                    song_link = customSearch.result()['result'][0]['link'].split("v=")[1]

                    print(song_link, current_song_id)

                frame = cv2.putText(frame, f"Next song: {song['name'].values[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                
            except Exception as e:
                pred_string = "No face"
                print(e)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    global current_song_id, duration, pred_string, song_link
    return render_template('index.html', current_song_id=current_song_id, token = access_token, duration = duration, pred_string= pred_string, song_link = song_link)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ws')
def ws():
    return socket.recv()

@app.route('/submit_playlist_url/<path:playlist_url>')
def submit_playlist_url(playlist_url):
    global mood_songs
    print(mood_songs.head())
    current_playlist_url = playlist_url

    print('Received Playlist URL:', current_playlist_url)
    playlist_id = current_playlist_url.split("playlist/")[1]
    playlist_id = playlist_id.split("?")[0]
    mood_songs = get_trending_playlist_data(playlist_id, access_token)
    print(mood_songs.head())
    return 'Playlist URL received'

@app.route('/get_current_song_id')
def get_current_song_id():
    return {'song_link': song_link}

if __name__ == '__main__':
    app.run(debug=True)
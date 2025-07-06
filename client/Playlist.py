import requests

class SpotifyPlaylist:
    def __init__(self, token):
        self.token = token
        self.api_url = 'https://api.spotify.com/v1/playlists'

    def get_playlist_info(self, playlist_id):
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        response = requests.get(f'{self.api_url}/{playlist_id}', headers=headers)
        response.raise_for_status()
        return response.json()
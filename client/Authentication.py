import requests
import base64

class SpotifyAuth:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = 'https://accounts.spotify.com/api/token'
        self.access_token = self.get_token()

    def get_token(self):
        headers = {
            'Authorization': 'Basic ' + base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        }
        data = {'grant_type': 'client_credentials'}
        response = requests.post(self.token_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()['access_token']
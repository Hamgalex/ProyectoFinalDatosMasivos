from client.Authentication import SpotifyAuth
from client.Playlist import SpotifyPlaylist

if __name__ == '__main__':
    client_id = 'uwu'
    client_secret = 'uwu'

    auth = SpotifyAuth(client_id, client_secret)
    playlist_api = SpotifyPlaylist(auth.access_token)

    playlist_id = '1puQ0hv40TUre24cFillJS'
    playlist = playlist_api.get_playlist_info(playlist_id)

    print(f"\n🎵 Playlist: {playlist['name']}")
    print(f"👤 Creada por: {playlist['owner']['display_name']}")
    print(f"🎧 Canciones:")
    for i, item in enumerate(playlist['tracks']['items']):
        track = item['track']
        print(f"{i+1}. {track['name']} - {track['artists'][0]['name']}")
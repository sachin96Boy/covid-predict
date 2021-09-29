import librosa
import wget

url = "https://www.python.org/static/img/python-logo@2x.png"

wget.download(url, 'c:/users/LikeGeeks/downloads/pythonLogo.png')
import requests


def download(url, path, chunk=2048):
    req = requests.get(url, stream=True)
    if req.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in req.iter_content(chunk):
                f.write(chunk)
            f.close()
        return path
    raise Exception('Given url is return status code:{}'.format(req.status_code))

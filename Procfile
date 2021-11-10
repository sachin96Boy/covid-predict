web: gunicorn sound_prediction.wsgi --log-file
web: python manage.py runsslserver 0.0.0.0:$PORT
web: gunicorn sound.wsgi -b 0.0.0.0:$PORT  --log-file 



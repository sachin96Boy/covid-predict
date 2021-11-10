web: gunicorn sound_prediction.wsgi --log-file
web: python manage.py runsslserver 0.0.0.0:$PORT
web: gunicorn -b 0.0.0.0:$PORT sound.wsgi:sound_predict  --log-file 



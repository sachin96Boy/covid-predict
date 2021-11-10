web: gunicorn sound_prediction.wsgi --log-file
web: gunicorn -b 0.0.0.0:$PORT sound.wsgi:application --log-file 
web: python manage.py runsslserver 0.0.0.0:$PORT



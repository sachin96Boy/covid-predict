web: python manage.py runserver 0.0.0.0:$PORT
worker: gunicorn -b 0.0.0.0:$PORT sound.wsgi:sound_predict  --log-file 



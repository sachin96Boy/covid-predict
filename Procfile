worker: gunicorn -b 0.0.0.0:$PORT sound.wsgi --log-file -
python manage.py collectstatic --noinput
manage.py migrate

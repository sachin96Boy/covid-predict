web: gunicorn -b 0.0.0.0:8000 sound.wsgi:application --log-file -
python manage.py collectstatic --noinput
manage.py migrate

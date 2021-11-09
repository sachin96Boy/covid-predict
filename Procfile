web: gunicorn -b 0.0.0.0:$PORT sound.wsgi:application --log-file -
python manage.py collectstatic --noinput
manage.py migrate
web:python manage.py runserver 0.0.0.0:$PORT

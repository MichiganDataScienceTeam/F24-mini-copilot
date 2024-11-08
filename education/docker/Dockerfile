FROM python:3.10.15-slim-bullseye

WORKDIR /app 

COPY requirements.txt requirements.txt

RUN pip install --compile --no-cache-dir -r requirements.txt

COPY basic_flask.py basic_flask.py

CMD ["flask", "--app", "basic_flask.py", "run", "-p", "8080"]




# Nicked from: https://github.com/markdouthwaite/streamlit-project/blob/master/Dockerfile
FROM python:3.9-slim

RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
EXPOSE 8080



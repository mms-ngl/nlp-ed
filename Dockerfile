FROM python:3.9-slim

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN apt-get update && apt-get install --no-install-recommends --yes build-essential
RUN pip install -r requirements.txt

# copy model

COPY model model

# copy code

COPY ed ed
ENV PYTHONPATH ed

# standard cmd

CMD [ "python", "ed/app.py" ]

FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# system u
RUN apt-get update

# install Python 3
RUN apt-get install -y \
    python3.10 \
    python3-pip 

# update PATH
ENV PATH="$PATH:/usr/bin/python3.10"

# create non-privilaged user
RUN addgroup --system fastapi && adduser --system --group fastapi fastapi
USER fastapi

# set program location
WORKDIR /app

# install Python dependencies
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --requirement requirements.txt

# copy source code
COPY . .

# port expose
EXPOSE 28000

# run application
CMD [ "fastapi", "run", "api.py", "--port", "28000" ]
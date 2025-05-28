FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# system u
RUN apt-get update

# install Python 3
RUN apt-get install -y \
    python3.10 \
    python3-pip 

# update PATH for Python
ENV PATH="$PATH:/usr/bin/python3.10"

# environment variables from GitHub Actions
ARG DATABASE_URI
ENV DATABASE_URI=$DATABASE_URI

# create non-privilaged user
RUN addgroup --system fastapi && adduser --system --group fastapi
USER fastapi

# set program location
WORKDIR /app

# install Python dependencies
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --requirement requirements.txt

# update PATH for fastapi
ENV PATH="${PATH}:/home/fastapi/.local/bin"

# copy source code
COPY . .

# port expose
EXPOSE 28000

# run application
CMD [ "fastapi", "run", "api.py", "--port", "28000" ]
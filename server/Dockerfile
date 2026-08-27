# The core path and nothing else. requirements.txt would pull silero-vad and
# faster-whisper - torch, gigabytes of it - for two fallbacks that a container
# with a Deepgram key never reaches, and the lock carries the test tools, which
# an image that is not running the tests has no use for. pgserver alone is a
# whole PostgreSQL, and there is a real one next door.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.lock ./
RUN grep -viE '^(pgserver|fakeredis|pytest|pytest-asyncio|iniconfig|pluggy|Pygments)' \
        requirements.lock > /tmp/runtime.txt \
    && pip install --no-cache-dir -r /tmp/runtime.txt

COPY . .

# Not root. This process decodes audio from whoever opened the socket, using a
# bundled ffmpeg, which is the last place to be running as anybody important.
RUN useradd --create-home app && chown -R app:app /app
USER app

# 0.0.0.0 because a container's own interface is the only thing this binds; the
# port reaches the outside when compose publishes it and not before. AUTH_TOKEN
# is still what stands between a published port and the internet - see Security.
ENV HOST=0.0.0.0 PYTHONUNBUFFERED=1
EXPOSE 8000

# Call records and, without DATABASE_URL, leads and bookings. kb.json ships in
# here too: a named volume is seeded from the image the first time it is used,
# so making the directory writable does not take the knowledge base away.
VOLUME ["/app/data"]

# /ready rather than /health: what compose and anything else waiting on this
# container want to know is whether it can take a call, and a worker that
# cannot reach its database cannot. urlopen raises on anything but a 2xx,
# which is exactly the exit code docker is asking for. No curl in slim.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c 'import urllib.request as u; u.urlopen("http://127.0.0.1:8000/ready")'

CMD ["python", "main.py"]

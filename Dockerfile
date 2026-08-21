# Digest-pinned, not tag-pinned. A floated base image changes the BLAS build,
# which changes float reduction order, which changes weights -- and a manifest
# whose code_digest no longer matches the weights it signed is worthless.
# Resolved from: python:3.11-slim, 2026-08-21.
FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

# Only settable before the interpreter starts; config/determinism.py refuses to
# run without it.
ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple -r requirements-dev.txt
COPY . .

CMD ["python", "scripts/spot_check_determinism.py"]

"""Stateless FastAPI app. Never trains -- no route imports engine.train or
engine.rebuild. If a route handler ever needs those, that is the gateway/worker
boundary breaking.
"""

from fastapi import FastAPI

from gateway.routes import erasure, models, predict

app = FastAPI(title="UnlearnShield Gateway")
app.include_router(erasure.router)
app.include_router(predict.router)
app.include_router(models.router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

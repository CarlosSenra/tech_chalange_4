from fastapi import FastAPI
from .routes import health

app = FastAPI(debug=True)

app.include_router(health.router)
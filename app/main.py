from fastapi import FastAPI
from app.routers import upload, highlight, player, frames

app = FastAPI(title="Basket Highlight AI", version="0.5.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(upload.router)
app.include_router(highlight.router)
app.include_router(player.router)
app.include_router(frames.router)

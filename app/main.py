import logging
from fastapi import FastAPI
from app.routes import router
from app.config import LOG_FILE

# Initialize FastAPI app
app = FastAPI()

# Include router
app.include_router(router)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@app.get("/")
async def health_check():
    """
    Endpoint to check if the server is running.
    """
    return {"message": "FastAPI server is running."}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI application
app = FastAPI(title="Bangla FastAPI Backend", version="1.0.0")

# Define the origins that are allowed to make cross-site requests to this API
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Apply CORS middleware layer
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

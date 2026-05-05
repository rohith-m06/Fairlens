from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Hello World"}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/diag")
async def diag():
    import os
    import sys
    return {
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "sys_path": sys.path
    }

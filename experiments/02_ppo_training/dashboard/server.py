from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os

app = FastAPI()

# Get absolute path to the directory
current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(current_dir, '..', 'metrics.csv')

# Serve static files for css and js
app.mount("/static", StaticFiles(directory=current_dir), name="static")

@app.get("/")
def read_root():
    with open(os.path.join(current_dir, 'index.html'), 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/api/metrics")
def get_metrics():
    if not os.path.exists(metrics_path):
        return JSONResponse(content={"error": "Metrics file not found"}, status_code=404)
    
    try:
        # Read metrics csv, filling NaN with None to be JSON compliant
        df = pd.read_csv(metrics_path)
        df = df.where(pd.notnull(df), None)
        # Convert to list of dicts
        records = df.to_dict(orient="records")
        return JSONResponse(content=records)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

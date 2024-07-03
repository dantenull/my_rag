import uvicorn
from main import app

try:
    uvicorn.run(app, host="0.0.0.0", port=8008)
except:
    pass

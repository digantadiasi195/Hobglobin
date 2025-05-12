#main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import os

app = FastAPI(title="Hobglobin RAG API")
rag = RAGPipeline()

class ChatRequest(BaseModel):
    query: str

# @app.get("/fine-prints")
# async def get_fine_prints():
#     try:
#         fine_prints = rag.generate_fine_prints()
#         return {"fine_prints": fine_prints}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = rag.chat(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
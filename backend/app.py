
from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import rag_chain
from fastapi.middleware.cors import CORSMiddleware  # For React cross-origin

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


@app.post("/chat")
async def chat(query: Query):
    try:
        response = rag_chain.run(query.query)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
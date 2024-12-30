from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest
from app.services.query_service import handle_query

router = APIRouter()

@router.post("/")
async def process_query(request: QueryRequest):
    try:
        result = handle_query(request.query)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, UploadFile
from app.services.document_service import process_file

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile):
    try:
        response = await process_file(file)
        return {"message": "File uploaded successfully", "data": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

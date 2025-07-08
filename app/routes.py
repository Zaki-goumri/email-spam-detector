from fastapi import APIRouter
from .model import EmailRequest, BatchEmailRequest
from .classifier import best_model

router = APIRouter()


@router.post("/predict")
def predictSpam(body: EmailRequest):
    pred = best_model.predict([body.message])[0]
    return {"result": pred}

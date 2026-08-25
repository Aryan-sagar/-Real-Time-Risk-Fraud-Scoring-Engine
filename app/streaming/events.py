from pydantic import BaseModel, Field


class TransactionEvent(BaseModel):
    transaction_id: str
    account_id: str
    amount: float = Field(gt=0)
    merchant: str
    merchant_base_risk: float = Field(ge=0, le=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: str
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

CaseStatus = Literal["open", "investigating", "closed"]
Feedback = Literal["fraud", "legitimate"]


class Transaction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    transaction_id: str = Field(min_length=1, max_length=128)
    TransactionAmt: float = Field(ge=0, allow_inf_nan=False)
    TransactionDT: int
    card1: Optional[int] = None
    addr1: Optional[int] = None
    DeviceInfo: Optional[str] = None
    P_emaildomain: Optional[str] = None
    account_id: Optional[str] = None
    card_id: Optional[str] = None
    device_id: Optional[str] = None
    extra_features: dict[str, Optional[Union[StrictBool, StrictInt, StrictFloat, StrictStr]]] = Field(default_factory=dict)


class CardHistory(BaseModel):
    transaction_count: int
    mean_amount: float


class FraudScore(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: int
    decision: str
    model: str
    card_history: Optional[CardHistory] = None
    gnn_score: Optional[float] = None
    xgb_probability: Optional[float] = None
    ring_score: Optional[float] = None
    ring_id: Optional[str] = None


class CaseUpdate(BaseModel):
    status: Optional[CaseStatus] = None
    label: Optional[Feedback] = None

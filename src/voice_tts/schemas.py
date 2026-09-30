from pydantic import BaseModel, Field


class PrepareReferenceRequest(BaseModel):
    source_name: str
    start_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=10, le=20)
    transcript: str


class SynthesisRequest(BaseModel):
    text: str

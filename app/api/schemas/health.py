from pydantic import BaseModel

class HealthOutput(BaseModel):

    healt: str = 'Ok!'
    status_code: int = 200
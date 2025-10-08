from pydantic import BaseModel, ConfigDict
from app.core.camel import to_camel

class CamelModel(BaseModel):
    # by_alias=True 로 직렬화, populate_by_name=True 로 snake/camel 둘 다 입력 허용
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
        ser_json_inf_nan=False,
    )

from pydantic import BaseModel
from pydantic.alias_generators import to_camel


# Pydantic 모델의 snake_case 필드명을
# 응답 시 자동으로 camelCase로 변환해주는 기본 모델
class CamelModel(BaseModel):
    """
    BaseModel that automatically converts field names from
    snake_case (internal Python) to camelCase (external JSON response).
    """

    class Config:
        # Pydantic v2에서는 model_config로 변경됨
        # model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
        
        # Pydantic v1 호환성 유지를 위해 Config 클래스 사용 (기존 코드 베이스 가정)
        allow_population_by_field_name = True
        alias_generator = to_camel
        
        # FastAPI의 response_model에 사용될 때 camelCase 적용을 위한 설정
        populate_by_name = True

    # Pydantic V2 Migration:
    # @model_validator(mode='before')
    # def check_any_unknown(cls, values):
    #     if isinstance(values, dict):
    #         # Ensures compatibility with json responses from pydantic.
    #         # Example: "userId" -> "user_id" on input
    #         return {to_snake_case(key): value for key, value in values.items()}
    #     return values
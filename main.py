from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncIterator, Callable
import logging

from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    InputFormat,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn


from enum import StrEnum, auto
from typing import Any
from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputFormat(StrEnum):
    MARKDOWN = auto()
    TEXT = auto()
    HTML = auto()


class ParseRequest(BaseRequest):
    include_json: bool = Field(
        False,
        description="Include a json representation of the document in the response",
    )
    output_format: OutputFormat = Field(
        OutputFormat.MARKDOWN, description="Output format of parsed text"
    )


class ParseUrlRequest(ParseRequest):
    url: str = Field(..., description="Download url for input file")


class ParseFileRequest(ParseRequest):
    @classmethod
    def from_form_data(
        cls,
        data: str = Form(..., examples=[ParseRequest().model_dump_json()]),
    ) -> "ParseFileRequest":
        return cls.model_validate_json(data)


class BaseResponse(BaseModel):
    message: str
    status: str


class ParseResponseData(BaseModel):
    output: str
    json_output: dict[str, Any] | None = None


class ParseResponse(BaseResponse):
    data: ParseResponseData


from pathlib import Path
import multiprocessing

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    num_workers: int | float | None = None
    log_level: str = "INFO"
    dev_mode: bool = False
    port: int = 8080
    auth_token: str | None = None
    ocr_languages: str = "en,es,fr,de,sv"
    do_code_enrichment: bool = False
    do_formula_enrichment: bool = False
    do_picture_classification: bool = False
    do_picture_description: bool = False

    def get_num_workers(self) -> int | None:
        if self.num_workers is None:
            return None

        if self.num_workers == -1:
            return multiprocessing.cpu_count()

        if 0 < self.num_workers < 1:
            return int(self.num_workers * multiprocessing.cpu_count())

        return int(self.num_workers)


def get_log_config(log_level: str):
    log_file = Path.cwd() / Path("logs/docling-inference.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "uvicorn": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": str(log_file),
                "mode": "a",
            },
            "uvicorn": {
                "formatter": "uvicorn",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default", "file"],
                "level": log_level,
            },
            "uvicorn": {
                "handlers": ["uvicorn"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["uvicorn"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["uvicorn"],
                "level": "INFO",
                "propagate": False,
            },
            "docling": {
                "handlers": ["default", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Setup and teardown events of the app"""
    # Setup
    config = Config()

    ocr_languages = config.ocr_languages.split(",")
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    ocr_options=EasyOcrOptions(lang=ocr_languages),
                    do_code_enrichment=False,
                    do_formula_enrichment=False,
                    do_picture_classification=False,
                    do_picture_description=False,
                )
            )
        }
    )
    for i, format in enumerate(InputFormat):
        logger.info(f"Initializing {format.value} pipeline {i + 1}/{len(InputFormat)}")

        converter.initialize_pipeline(format)

    app.state.converter = converter
    app.state.config = config

    yield
    # Teardown


app = FastAPI(lifespan=lifespan)

bearer_auth = HTTPBearer(auto_error=False)


async def authorize_header(
    request: Request, bearer: HTTPAuthorizationCredentials | None = Depends(bearer_auth)
) -> None:
    # Do nothing if AUTH_KEY is not set
    auth_token: str | None = request.app.state.config.auth_token
    if auth_token is None:
        return

    # Validate auth bearer
    if bearer is None or bearer.credentials != auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Unauthorized"},
        )


@app.exception_handler(Exception)
async def ingestion_error_handler(_, exc: Exception) -> None:
    detail = {"message": str(exc)}
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
    )


ConvertData = str | Path | DocumentStream
ConvertFunc = Callable[[ConvertData], ConversionResult]


def convert(request: Request) -> ConvertFunc:
    def convert_func(data: ConvertData) -> ConversionResult:
        try:
            result = request.app.state.converter.convert(data, raises_on_error=False)
            _check_conversion_result(result)
            return result
        except FileNotFoundError as exc:
            logger.error(f"File not found error: {str(exc)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Input not found"},
            ) from exc

    return convert_func


@app.post("/parse/url", response_model=ParseResponse)
def parse_document_url(
    payload: ParseUrlRequest,
    convert: ConvertFunc = Depends(convert),
    _=Depends(authorize_header),
) -> ParseResponse:
    result = convert(payload.url)
    output = _get_output(result.document, payload.output_format)

    json_output = result.document.export_to_dict() if payload.include_json else None

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
        data=ParseResponseData(output=output, json_output=json_output),
    )


@app.post("/parse/file", response_model=ParseResponse)
def parse_document_stream(
    file: UploadFile,
    convert: ConvertFunc = Depends(convert),
    payload: ParseFileRequest = Depends(ParseFileRequest.from_form_data),
    _=Depends(authorize_header),
) -> ParseResponse:
    binary_data = file.file.read()
    data = DocumentStream(
        name=file.filename or "unset_name", stream=BytesIO(binary_data)
    )

    result = convert(data)
    output = _get_output(result.document, payload.output_format)

    json_output = result.document.export_to_dict() if payload.include_json else None

    return ParseResponse(
        message="Document parsed successfully",
        status="Ok",
        data=ParseResponseData(output=output, json_output=json_output),
    )


def _check_conversion_result(result: ConversionResult) -> None:
    """Raises HTTPException and logs on error"""
    if result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
        return

    if result.errors:
        for error in result.errors:
            if error.component_type == DoclingComponentType.USER_INPUT:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"message": error.error_message},
                )
            logger.error(
                f"Error in: {error.component_type.name} - {error.error_message}"
            )
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _get_output(document: DoclingDocument, format: OutputFormat) -> str:
    if format == OutputFormat.MARKDOWN:
        return document.export_to_markdown()
    if format == OutputFormat.TEXT:
        return document.export_to_text()
    if format == OutputFormat.HTML:
        return document.export_to_html()


if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        log_config=get_log_config(config.log_level),
        reload=config.dev_mode,
        workers=config.get_num_workers(),
    )

from fastapi import Request
from fastapi.responses import JSONResponse
from app.utils.logger import log


class AppException(Exception):
    def __init__(self, code: int = 500, message: str = "Internal error", detail: str = ""):
        self.code = code
        self.message = message
        self.detail = detail


class NotFoundException(AppException):
    def __init__(self, resource: str = "Resource"):
        super().__init__(code=404, message=f"{resource} not found")


class BadRequestException(AppException):
    def __init__(self, detail: str = "Bad request"):
        super().__init__(code=400, message="Bad request", detail=detail)


async def app_exception_handler(request: Request, exc: AppException):
    log.warning(f"AppException: {exc.code} {exc.message} - {exc.detail}")
    return JSONResponse(
        status_code=exc.code,
        content={"code": exc.code, "message": exc.message, "detail": exc.detail},
    )


async def generic_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    log.error(f"Unhandled exception: {type(exc).__name__}: {exc}\n{''.join(tb)}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "Internal server error", "detail": str(exc)},
    )

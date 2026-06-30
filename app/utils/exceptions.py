from fastapi import Request
from fastapi.responses import JSONResponse
from app.utils.logger import log


async def generic_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    log.error(f"Unhandled exception: {type(exc).__name__}: {exc}\n{''.join(tb)}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "Internal server error", "detail": str(exc)},
    )

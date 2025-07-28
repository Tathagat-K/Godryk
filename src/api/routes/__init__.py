from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get('/health')
def health_check():
    logger.debug("Health check endpoint called")
    return {'status': 'ok'}

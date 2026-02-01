"""
Job queue setup using arq (async Redis queue).
"""
import logging
from typing import Optional
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from ..config import Config

logger = logging.getLogger(__name__)
config = Config()

# Redis settings for arq
ARQ_REDIS_SETTINGS = RedisSettings(
    host=config.redis_host,
    port=config.redis_port,
    database=0
)


class JobQueue:
    """Wrapper for arq job queue operations."""

    _pool: Optional[ArqRedis] = None

    @classmethod
    async def initialize(cls):
        """Initialize the job queue by creating the Redis pool."""
        await cls.get_pool()
        logger.info(f"Job queue initialized: {config.redis_host}:{config.redis_port}")

    @classmethod
    async def close(cls):
        """Close the job queue connection."""
        await cls.close_pool()

    @classmethod
    async def get_pool(cls) -> ArqRedis:
        """Get or create the Redis connection pool."""
        if cls._pool is None:
            cls._pool = await create_pool(ARQ_REDIS_SETTINGS)
            logger.info(f"Created arq Redis pool: {config.redis_host}:{config.redis_port}")
        return cls._pool

    @classmethod
    async def close_pool(cls):
        """Close the Redis connection pool."""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("Closed arq Redis pool")

    @classmethod
    async def enqueue_job(cls, function_name: str, *args, **kwargs) -> str:
        """
        Enqueue a job to be processed by workers.

        Args:
            function_name: Name of the worker function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            job_id: Unique ID for the enqueued job
        """
        pool = await cls.get_pool()
        job = await pool.enqueue_job(function_name, *args, **kwargs)
        logger.info(f"Enqueued job {job.job_id}: {function_name}")
        return job.job_id

    @classmethod
    async def get_job_result(cls, job_id: str):
        """Get the result of a completed job."""
        pool = await cls.get_pool()
        job = await pool.job(job_id)
        if job:
            return await job.result()
        return None

    @classmethod
    async def get_job_status(cls, job_id: str) -> Optional[str]:
        """Get the status of a job."""
        pool = await cls.get_pool()
        job = await pool.job(job_id)
        if job:
            return await job.status()
        return None

    @classmethod
    async def check_health(cls) -> dict:
        """Check Redis connectivity."""
        pool = await cls.get_pool()
        # Test Redis connection with PING command
        result = await pool.ping()
        return {"status": "healthy" if result else "unhealthy"}

"""
api/config.py
=============
Application settings loaded from environment variables / .env file.
All sensitive values (API keys, secrets) must come from the environment —
never hard-coded.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # VirusTotal (optional — empty string means demo/mock mode)
    vt_api_key: str = ""

    # CORS — comma-separated origins, e.g. "https://app.example.com"
    allowed_origins: str = "*"

    # Rate limiting — max requests per minute per IP (0 = disabled)
    rate_limit: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse the comma-separated ALLOWED_ORIGINS string into a list."""
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton settings instance."""
    return Settings()

"""Configuration settings manager.

Handles loading and validation of configuration using Pydantic with support
for environment variables and secrets management.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum
import yaml
import json
from pydantic import BaseModel, Field, validator, BaseSettings
from pydantic.env_settings import SettingsSourceCallable
import boto3
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from google.cloud import secretmanager
from hydrobot.utils.logger_setup import get_logger

logger = get_logger(__name__)

class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'

class ExchangeSettings(BaseSettings):
    """Exchange configuration settings."""
    name: str
    trading_pairs: List[str]
    testnet: bool = False
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit_margin: float = 0.8

class TradingSettings(BaseSettings):
    """Trading parameters."""
    base_order_size: float
    max_position_size: float
    min_spread: float = 0.001
    max_slippage: float = 0.002
    order_timeout: int = 5
    position_timeout: int = 300
    min_profit_threshold: float = 0.001

class RiskSettings(BaseSettings):
    """Risk management settings."""
    max_drawdown: float = Field(..., gt=0, lt=1)
    daily_loss_limit: float = Field(..., gt=0, lt=1)
    error_threshold: int = 3
    circuit_breaker_cooldown: int = 900
    position_limits: Dict[str, float]

class ModelSettings(BaseSettings):
    """ML model settings."""
    feature_window: int = 100
    prediction_horizon: int = 10
    min_samples: int = 1000
    training_interval: int = 86400
    save_path: str = "models"

class MonitoringSettings(BaseSettings):
    """Monitoring configuration."""
    metrics_port: int = 8000
    push_gateway: Optional[str] = None
    alert_webhook: Optional[str] = None

class SecretsManager:
    """Cloud-based secrets management."""
    
    @staticmethod
    async def get_aws_secret(secret_name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except Exception as e:
            logger.error("aws_secret_error", error=str(e))
            return None
    
    @staticmethod
    async def get_azure_secret(vault_name: str, secret_name: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        try:
            credential = DefaultAzureCredential()
            client = SecretClient(
                vault_url=f"https://{vault_name}.vault.azure.net/",
                credential=credential
            )
            secret = client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error("azure_secret_error", error=str(e))
            return None
    
    @staticmethod
    async def get_gcp_secret(project_id: str, secret_name: str) -> Optional[str]:
        """Get secret from Google Cloud Secret Manager."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error("gcp_secret_error", error=str(e))
            return None

class Settings(BaseSettings):
    """Main application settings."""
    env: Environment = Environment.PRODUCTION
    log_level: str = "INFO"
    
    # Component settings
    exchange: ExchangeSettings
    trading: TradingSettings
    risk: RiskSettings
    models: ModelSettings
    monitoring: MonitoringSettings
    
    # Cloud settings
    aws_secret_name: Optional[str] = None
    azure_vault_name: Optional[str] = None
    gcp_project_id: Optional[str] = None
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        config_base_path = Path(__file__).parent.parent.parent
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            def yaml_config_source_inner(settings: BaseSettings) -> Dict[str, Any]:
                encoding = settings.__config__.env_file_encoding
                yaml_path = cls.config_base_path / "hydrobot" / "config" / "config.yaml"
                
                if yaml_path.exists():
                    logger.info(f"Loading configuration from: {yaml_path}")
                    try:
                        return yaml.safe_load(yaml_path.read_text(encoding))
                    except Exception as e:
                        logger.error(f"Error loading YAML config from {yaml_path}: {e}")
                        return {}
                else:
                    logger.warning(f"YAML config file not found at: {yaml_path}")
                    return {}

            return (
                init_settings,
                yaml_config_source_inner,  # Load from YAML first
                env_settings,  # Then environment variables
                file_secret_settings,  # Finally .env file
            )
    
    async def load_secrets(self) -> None:
        """Load secrets from cloud providers or .env file."""
        logger.info("Attempting to load secrets...")
        loaded_from_cloud = False
        
        if self.aws_secret_name:
            logger.info(f"Attempting to load secrets from AWS Secrets Manager: {self.aws_secret_name}")
            secret_dict = await SecretsManager.get_aws_secret(self.aws_secret_name)
            if secret_dict:
                try:
                    secrets = json.loads(secret_dict)
                    self.exchange.api_key = secrets.get('EXCHANGE_API_KEY', self.exchange.api_key)
                    self.exchange.api_secret = secrets.get('EXCHANGE_SECRET_KEY', self.exchange.api_secret)
                    logger.info("Successfully loaded secrets from AWS.")
                    loaded_from_cloud = True
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON secret from AWS.")
                except Exception as e:
                    logger.error(f"Error processing AWS secret: {e}")

        elif self.azure_vault_name:
            logger.info(f"Attempting to load secrets from Azure Key Vault: {self.azure_vault_name}")
            key = await SecretsManager.get_azure_secret(self.azure_vault_name, 'EXCHANGE-API-KEY')
            secret = await SecretsManager.get_azure_secret(self.azure_vault_name, 'EXCHANGE-SECRET-KEY')
            if key:
                self.exchange.api_key = key
                loaded_from_cloud = True
            if secret:
                self.exchange.api_secret = secret
                loaded_from_cloud = True
            if loaded_from_cloud:
                logger.info("Successfully loaded secrets from Azure Key Vault.")
            else:
                logger.warning("Failed to load secrets from Azure Key Vault.")

        elif self.gcp_project_id:
            logger.info(f"Attempting to load secrets from GCP Secret Manager, project: {self.gcp_project_id}")
            key = await SecretsManager.get_gcp_secret(self.gcp_project_id, 'exchange-api-key')
            secret = await SecretsManager.get_gcp_secret(self.gcp_project_id, 'exchange-secret-key')
            if key:
                self.exchange.api_key = key
                loaded_from_cloud = True
            if secret:
                self.exchange.api_secret = secret
                loaded_from_cloud = True
            if loaded_from_cloud:
                logger.info("Successfully loaded secrets from GCP.")
            else:
                logger.warning("Failed to load secrets from GCP.")

        if not loaded_from_cloud:
            logger.info("Secrets not loaded from cloud providers. Relying on environment variables or .env file.")
            if not self.exchange.api_key:
                logger.warning("Exchange API Key is missing after checking cloud and environment/.env.")
            if not self.exchange.api_secret:
                logger.warning("Exchange API Secret is missing after checking cloud and environment/.env.")
        
        key_status = "present" if self.exchange.api_key else "missing"
        secret_status = "present" if self.exchange.api_secret else "missing"
        logger.info(f"Secret loading complete. API Key: {key_status}, API Secret: {secret_status}")

    @validator('trading')
    def validate_trading_params(cls, v):
        """Validate trading parameters."""
        if v.base_order_size <= 0:
            raise ValueError("base_order_size must be positive")
        if v.max_position_size < v.base_order_size:
            raise ValueError("max_position_size must be >= base_order_size")
        return v
    
    @validator('risk')
    def validate_risk_params(cls, v):
        """Validate risk parameters."""
        if not 0 < v.max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if not 0 < v.daily_loss_limit < 1:
            raise ValueError("daily_loss_limit must be between 0 and 1")
        return v

try:
    settings = Settings()
except Exception as e:
    import logging as std_logging
    std_logging.basicConfig(level="ERROR")
    std_logging.error(f"CRITICAL: Failed to initialize settings: {e}", exc_info=True)
    raise SystemExit(f"Failed to initialize settings: {e}")

async def load_secrets_into_settings():
    """Async function to load secrets into the global settings object."""
    await settings.load_secrets()
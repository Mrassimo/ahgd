"""
AHGD Secrets Management System

This module provides comprehensive secrets management with support for:
- Environment variables
- Secure key handling
- AWS Secrets Manager integration
- Azure Key Vault integration
- Local encrypted secrets storage
- Configuration encryption support
- Secrets rotation and caching
"""

import os
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading
import time

# Optional imports for cloud providers
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretProvider(Enum):
    """Supported secret providers"""
    ENVIRONMENT = "environment"
    FILE = "file"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    ENCRYPTED_FILE = "encrypted_file"


@dataclass
class SecretMetadata:
    """Metadata for a secret"""
    key: str
    provider: SecretProvider
    created_at: datetime
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    tags: Dict[str, str] = None
    description: str = ""


class SecretNotFoundError(Exception):
    """Raised when a secret is not found"""
    pass


class SecretProviderError(Exception):
    """Raised when there's an error with a secret provider"""
    pass


class EncryptionError(Exception):
    """Raised when there's an encryption/decryption error"""
    pass


class SecretCache:
    """Thread-safe cache for secrets with TTL support"""
    
    def __init__(self, default_ttl_seconds: int = 300):  # 5 minutes default
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if datetime.now() > entry['expires_at']:
                del self._cache[key]
                return None
            
            entry['last_accessed'] = datetime.now()
            return entry['value']
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache with TTL"""
        ttl = ttl_seconds or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        with self._lock:
            self._cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'last_accessed': datetime.now()
            }
    
    def remove(self, key: str):
        """Remove key from cache"""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cached values"""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry['expires_at']
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            now = datetime.now()
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values()
                if now > entry['expires_at']
            )
            
            return {
                'total_entries': total_entries,
                'active_entries': total_entries - expired_entries,
                'expired_entries': expired_entries,
                'cache_hit_potential': (total_entries - expired_entries) / max(total_entries, 1)
            }


class LocalEncryption:
    """Local encryption utilities for sensitive data"""
    
    def __init__(self, key_path: Optional[Path] = None):
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required for encryption features")
        
        self.key_path = key_path or Path.home() / '.ahgd' / 'encryption.key'
        self._fernet = None
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        if self.key_path.exists():
            return self.key_path.read_bytes()
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_bytes(key)
            self.key_path.chmod(0o600)  # Restrict permissions
            logger.info(f"Generated new encryption key: {self.key_path}")
            return key
    
    def _get_fernet(self) -> Fernet:
        """Get Fernet encryption instance"""
        if self._fernet is None:
            key = self._get_or_create_key()
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self._get_fernet().encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded encrypted data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self._get_fernet().decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt data: {e}")
    
    def rotate_key(self) -> str:
        """Rotate encryption key and return old key for migration"""
        old_key = None
        if self.key_path.exists():
            old_key = self.key_path.read_text()
        
        # Generate new key
        new_key = Fernet.generate_key()
        self.key_path.write_bytes(new_key)
        self._fernet = None  # Reset cached instance
        
        logger.info("Encryption key rotated")
        return old_key


class AWSSecretsProvider:
    """AWS Secrets Manager provider"""
    
    def __init__(self, region_name: str = None):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 library required for AWS Secrets Manager")
        
        self.region_name = region_name or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self._client = None
    
    def _get_client(self):
        """Get AWS Secrets Manager client"""
        if self._client is None:
            try:
                self._client = boto3.client('secretsmanager', region_name=self.region_name)
            except NoCredentialsError:
                raise SecretProviderError("AWS credentials not configured")
        return self._client
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from AWS Secrets Manager"""
        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                raise SecretNotFoundError(f"Secret '{secret_name}' not found in AWS Secrets Manager")
            else:
                raise SecretProviderError(f"AWS Secrets Manager error: {e}")
    
    def set_secret(self, secret_name: str, secret_value: str, description: str = "") -> bool:
        """Create or update secret in AWS Secrets Manager"""
        try:
            client = self._get_client()
            
            # Try to update first
            try:
                client.update_secret(
                    SecretId=secret_name,
                    SecretString=secret_value,
                    Description=description
                )
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new secret
                    client.create_secret(
                        Name=secret_name,
                        SecretString=secret_value,
                        Description=description
                    )
                    return True
                else:
                    raise
        except ClientError as e:
            raise SecretProviderError(f"Failed to set AWS secret: {e}")
    
    def list_secrets(self) -> List[str]:
        """List all secrets in AWS Secrets Manager"""
        try:
            client = self._get_client()
            response = client.list_secrets()
            return [secret['Name'] for secret in response['SecretList']]
        except ClientError as e:
            raise SecretProviderError(f"Failed to list AWS secrets: {e}")


class AzureKeyVaultProvider:
    """Azure Key Vault provider"""
    
    def __init__(self, vault_url: str = None):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-keyvault-secrets library required for Azure Key Vault")
        
        self.vault_url = vault_url or os.getenv('AZURE_KEY_VAULT_URL')
        if not self.vault_url:
            raise SecretProviderError("Azure Key Vault URL not configured")
        
        self._client = None
    
    def _get_client(self):
        """Get Azure Key Vault client"""
        if self._client is None:
            try:
                credential = DefaultAzureCredential()
                self._client = SecretClient(vault_url=self.vault_url, credential=credential)
            except Exception as e:
                raise SecretProviderError(f"Failed to initialize Azure Key Vault client: {e}")
        return self._client
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault"""
        try:
            client = self._get_client()
            secret = client.get_secret(secret_name)
            return secret.value
        except ResourceNotFoundError:
            raise SecretNotFoundError(f"Secret '{secret_name}' not found in Azure Key Vault")
        except Exception as e:
            raise SecretProviderError(f"Azure Key Vault error: {e}")
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Create or update secret in Azure Key Vault"""
        try:
            client = self._get_client()
            client.set_secret(secret_name, secret_value)
            return True
        except Exception as e:
            raise SecretProviderError(f"Failed to set Azure secret: {e}")
    
    def list_secrets(self) -> List[str]:
        """List all secrets in Azure Key Vault"""
        try:
            client = self._get_client()
            secrets = client.list_properties_of_secrets()
            return [secret.name for secret in secrets]
        except Exception as e:
            raise SecretProviderError(f"Failed to list Azure secrets: {e}")


class SecretsManager:
    """
    Comprehensive secrets management system with multiple provider support,
    caching, and encryption capabilities.
    """
    
    def __init__(
        self,
        secrets_dir: Union[str, Path] = None,
        cache_ttl_seconds: int = 300,
        enable_cache: bool = True,
        enable_encryption: bool = True,
        providers: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.secrets_dir = Path(secrets_dir) if secrets_dir else Path.home() / '.ahgd' / 'secrets'
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache setup
        self.cache = SecretCache(cache_ttl_seconds) if enable_cache else None
        
        # Encryption setup
        self.encryption = LocalEncryption() if enable_encryption and CRYPTO_AVAILABLE else None
        
        # Provider setup
        self.providers: Dict[SecretProvider, Any] = {}
        self._setup_providers(providers or {})
        
        # Metadata tracking
        self._metadata: Dict[str, SecretMetadata] = {}
        self._load_metadata()
        
        logger.info(f"Secrets manager initialized with {len(self.providers)} providers")
    
    def _setup_providers(self, provider_configs: Dict[str, Dict[str, Any]]):
        """Setup configured secret providers"""
        # Always available: environment and file
        self.providers[SecretProvider.ENVIRONMENT] = None
        self.providers[SecretProvider.FILE] = None
        
        if self.encryption:
            self.providers[SecretProvider.ENCRYPTED_FILE] = self.encryption
        
        # AWS Secrets Manager
        if 'aws' in provider_configs and AWS_AVAILABLE:
            try:
                aws_config = provider_configs['aws']
                self.providers[SecretProvider.AWS_SECRETS_MANAGER] = AWSSecretsProvider(
                    region_name=aws_config.get('region')
                )
                logger.info("AWS Secrets Manager provider enabled")
            except Exception as e:
                logger.warning(f"Failed to setup AWS provider: {e}")
        
        # Azure Key Vault
        if 'azure' in provider_configs and AZURE_AVAILABLE:
            try:
                azure_config = provider_configs['azure']
                self.providers[SecretProvider.AZURE_KEY_VAULT] = AzureKeyVaultProvider(
                    vault_url=azure_config.get('vault_url')
                )
                logger.info("Azure Key Vault provider enabled")
            except Exception as e:
                logger.warning(f"Failed to setup Azure provider: {e}")
    
    def _load_metadata(self):
        """Load secrets metadata from disk"""
        metadata_file = self.secrets_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    for key, meta_dict in data.items():
                        self._metadata[key] = SecretMetadata(
                            key=meta_dict['key'],
                            provider=SecretProvider(meta_dict['provider']),
                            created_at=datetime.fromisoformat(meta_dict['created_at']),
                            last_accessed=datetime.fromisoformat(meta_dict['last_accessed']) if meta_dict.get('last_accessed') else None,
                            expires_at=datetime.fromisoformat(meta_dict['expires_at']) if meta_dict.get('expires_at') else None,
                            tags=meta_dict.get('tags', {}),
                            description=meta_dict.get('description', '')
                        )
            except Exception as e:
                logger.warning(f"Failed to load secrets metadata: {e}")
    
    def _save_metadata(self):
        """Save secrets metadata to disk"""
        metadata_file = self.secrets_dir / 'metadata.json'
        try:
            data = {}
            for key, metadata in self._metadata.items():
                data[key] = {
                    'key': metadata.key,
                    'provider': metadata.provider.value,
                    'created_at': metadata.created_at.isoformat(),
                    'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                    'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                    'tags': metadata.tags or {},
                    'description': metadata.description
                }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save secrets metadata: {e}")
    
    def _update_metadata(self, key: str, provider: SecretProvider, description: str = ""):
        """Update metadata for a secret"""
        now = datetime.now()
        if key in self._metadata:
            self._metadata[key].last_accessed = now
        else:
            self._metadata[key] = SecretMetadata(
                key=key,
                provider=provider,
                created_at=now,
                last_accessed=now,
                description=description
            )
        self._save_metadata()
    
    def get_secret(self, key: str, default: Optional[str] = None, providers: Optional[List[SecretProvider]] = None) -> str:
        """
        Get secret value from the first available provider.
        
        Args:
            key: Secret key/name
            default: Default value if secret not found
            providers: List of providers to try (in order)
        
        Returns:
            Secret value
        
        Raises:
            SecretNotFoundError: If secret not found and no default provided
        """
        # Check cache first
        if self.cache:
            cached_value = self.cache.get(key)
            if cached_value is not None:
                self._update_metadata(key, SecretProvider.ENVIRONMENT)  # Provider unknown for cached
                return cached_value
        
        # Determine provider order
        provider_order = providers or [
            SecretProvider.ENVIRONMENT,
            SecretProvider.ENCRYPTED_FILE,
            SecretProvider.AWS_SECRETS_MANAGER,
            SecretProvider.AZURE_KEY_VAULT,
            SecretProvider.FILE
        ]
        
        last_error = None
        
        for provider in provider_order:
            if provider not in self.providers:
                continue
            
            try:
                value = self._get_from_provider(key, provider)
                if value is not None:
                    # Cache the result
                    if self.cache:
                        self.cache.set(key, value)
                    
                    # Update metadata
                    self._update_metadata(key, provider)
                    
                    return value
            except SecretNotFoundError:
                continue
            except Exception as e:
                last_error = e
                logger.warning(f"Error getting secret '{key}' from {provider.value}: {e}")
                continue
        
        # Secret not found in any provider
        if default is not None:
            return default
        
        if last_error:
            raise SecretProviderError(f"Failed to get secret '{key}': {last_error}")
        else:
            raise SecretNotFoundError(f"Secret '{key}' not found in any provider")
    
    def _get_from_provider(self, key: str, provider: SecretProvider) -> Optional[str]:
        """Get secret from specific provider"""
        if provider == SecretProvider.ENVIRONMENT:
            return self._get_from_environment(key)
        elif provider == SecretProvider.FILE:
            return self._get_from_file(key)
        elif provider == SecretProvider.ENCRYPTED_FILE:
            return self._get_from_encrypted_file(key)
        elif provider == SecretProvider.AWS_SECRETS_MANAGER:
            return self.providers[provider].get_secret(key)
        elif provider == SecretProvider.AZURE_KEY_VAULT:
            return self.providers[provider].get_secret(key)
        else:
            raise SecretProviderError(f"Unknown provider: {provider}")
    
    def _get_from_environment(self, key: str) -> Optional[str]:
        """Get secret from environment variables"""
        # Try multiple environment variable formats
        env_keys = [
            key,
            key.upper(),
            f"AHGD_{key.upper()}",
            f"SECRET_{key.upper()}",
            key.replace('-', '_').upper(),
            key.replace('.', '_').upper()
        ]
        
        for env_key in env_keys:
            value = os.getenv(env_key)
            if value is not None:
                return value
        
        return None
    
    def _get_from_file(self, key: str) -> Optional[str]:
        """Get secret from file"""
        file_path = self.secrets_dir / f"{key}.txt"
        if file_path.exists():
            return file_path.read_text().strip()
        return None
    
    def _get_from_encrypted_file(self, key: str) -> Optional[str]:
        """Get secret from encrypted file"""
        if not self.encryption:
            return None
        
        file_path = self.secrets_dir / f"{key}.enc"
        if file_path.exists():
            encrypted_data = file_path.read_text().strip()
            return self.encryption.decrypt(encrypted_data)
        return None
    
    def set_secret(
        self,
        key: str,
        value: str,
        provider: SecretProvider = SecretProvider.ENCRYPTED_FILE,
        description: str = "",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set secret in specified provider.
        
        Args:
            key: Secret key/name
            value: Secret value
            provider: Provider to store in
            description: Optional description
            ttl_seconds: Time-to-live for cache
        
        Returns:
            True if successful
        """
        try:
            if provider == SecretProvider.FILE:
                file_path = self.secrets_dir / f"{key}.txt"
                file_path.write_text(value)
                file_path.chmod(0o600)
            
            elif provider == SecretProvider.ENCRYPTED_FILE:
                if not self.encryption:
                    raise SecretProviderError("Encryption not available")
                file_path = self.secrets_dir / f"{key}.enc"
                encrypted_value = self.encryption.encrypt(value)
                file_path.write_text(encrypted_value)
                file_path.chmod(0o600)
            
            elif provider == SecretProvider.AWS_SECRETS_MANAGER:
                if provider not in self.providers:
                    raise SecretProviderError("AWS Secrets Manager not configured")
                return self.providers[provider].set_secret(key, value, description)
            
            elif provider == SecretProvider.AZURE_KEY_VAULT:
                if provider not in self.providers:
                    raise SecretProviderError("Azure Key Vault not configured")
                return self.providers[provider].set_secret(key, value)
            
            else:
                raise SecretProviderError(f"Cannot set secret in provider: {provider}")
            
            # Update cache
            if self.cache:
                self.cache.set(key, value, ttl_seconds)
            
            # Update metadata
            self._update_metadata(key, provider, description)
            
            logger.info(f"Secret '{key}' stored in {provider.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set secret '{key}': {e}")
            raise SecretProviderError(f"Failed to set secret: {e}")
    
    def delete_secret(self, key: str, provider: Optional[SecretProvider] = None) -> bool:
        """Delete secret from provider(s)"""
        deleted = False
        
        providers_to_check = [provider] if provider else list(self.providers.keys())
        
        for prov in providers_to_check:
            try:
                if prov == SecretProvider.FILE:
                    file_path = self.secrets_dir / f"{key}.txt"
                    if file_path.exists():
                        file_path.unlink()
                        deleted = True
                
                elif prov == SecretProvider.ENCRYPTED_FILE:
                    file_path = self.secrets_dir / f"{key}.enc"
                    if file_path.exists():
                        file_path.unlink()
                        deleted = True
                
                # Cloud providers would need delete methods implemented
                
            except Exception as e:
                logger.warning(f"Failed to delete secret '{key}' from {prov.value}: {e}")
        
        if deleted:
            # Remove from cache
            if self.cache:
                self.cache.remove(key)
            
            # Remove metadata
            self._metadata.pop(key, None)
            self._save_metadata()
            
            logger.info(f"Secret '{key}' deleted")
        
        return deleted
    
    def list_secrets(self, provider: Optional[SecretProvider] = None) -> List[str]:
        """List all secrets from provider(s)"""
        secrets = set()
        
        providers_to_check = [provider] if provider else list(self.providers.keys())
        
        for prov in providers_to_check:
            try:
                if prov == SecretProvider.FILE:
                    for file_path in self.secrets_dir.glob("*.txt"):
                        secrets.add(file_path.stem)
                
                elif prov == SecretProvider.ENCRYPTED_FILE:
                    for file_path in self.secrets_dir.glob("*.enc"):
                        secrets.add(file_path.stem)
                
                elif prov == SecretProvider.AWS_SECRETS_MANAGER and prov in self.providers:
                    aws_secrets = self.providers[prov].list_secrets()
                    secrets.update(aws_secrets)
                
                elif prov == SecretProvider.AZURE_KEY_VAULT and prov in self.providers:
                    azure_secrets = self.providers[prov].list_secrets()
                    secrets.update(azure_secrets)
                
            except Exception as e:
                logger.warning(f"Failed to list secrets from {prov.value}: {e}")
        
        return sorted(list(secrets))
    
    def get_metadata(self, key: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret"""
        return self._metadata.get(key)
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        if self.cache:
            self.cache.cleanup_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get secrets manager statistics"""
        stats = {
            'total_secrets': len(self._metadata),
            'providers_configured': len(self.providers),
            'providers_available': [p.value for p in self.providers.keys()],
            'secrets_dir': str(self.secrets_dir),
            'encryption_enabled': self.encryption is not None,
            'cache_enabled': self.cache is not None
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        return stats
    
    def rotate_encryption_key(self) -> bool:
        """Rotate local encryption key and re-encrypt all encrypted secrets"""
        if not self.encryption:
            return False
        
        try:
            # Get all encrypted secrets
            encrypted_secrets = {}
            for file_path in self.secrets_dir.glob("*.enc"):
                key = file_path.stem
                encrypted_secrets[key] = self.get_secret(key)
            
            # Rotate key
            old_key = self.encryption.rotate_key()
            
            # Re-encrypt all secrets with new key
            for key, value in encrypted_secrets.items():
                self.set_secret(key, value, SecretProvider.ENCRYPTED_FILE)
            
            logger.info(f"Rotated encryption key for {len(encrypted_secrets)} secrets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate encryption key: {e}")
            return False


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(**kwargs) -> SecretsManager:
    """Get or create global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(**kwargs)
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None) -> str:
    """Convenience function to get a secret"""
    return get_secrets_manager().get_secret(key, default)


def set_secret(key: str, value: str, **kwargs) -> bool:
    """Convenience function to set a secret"""
    return get_secrets_manager().set_secret(key, value, **kwargs)


def delete_secret(key: str, **kwargs) -> bool:
    """Convenience function to delete a secret"""
    return get_secrets_manager().delete_secret(key, **kwargs)


def list_secrets(**kwargs) -> List[str]:
    """Convenience function to list secrets"""
    return get_secrets_manager().list_secrets(**kwargs)


# Decorators for secret injection
def requires_secret(secret_key: str, default: Optional[str] = None):
    """Decorator to ensure a secret is available"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                secret_value = get_secret(secret_key, default)
                return func(*args, **kwargs, **{f'{secret_key}_secret': secret_value})
            except SecretNotFoundError:
                raise SecretNotFoundError(f"Required secret '{secret_key}' not found")
        return wrapper
    return decorator


def with_secret(secret_key: str, default: Optional[str] = None):
    """Decorator to inject secret as function argument"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            secret_value = get_secret(secret_key, default)
            return func(*args, secret_value, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # CLI usage for testing secrets
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        secrets_manager = get_secrets_manager()
        
        if command == "get":
            if len(sys.argv) < 3:
                print("Usage: python secrets.py get <key>")
                sys.exit(1)
            key = sys.argv[2]
            try:
                value = secrets_manager.get_secret(key)
                print(f"Secret '{key}': {value}")
            except SecretNotFoundError:
                print(f"Secret '{key}' not found")
                sys.exit(1)
        
        elif command == "set":
            if len(sys.argv) < 4:
                print("Usage: python secrets.py set <key> <value>")
                sys.exit(1)
            key = sys.argv[2]
            value = sys.argv[3]
            if secrets_manager.set_secret(key, value):
                print(f"Secret '{key}' set successfully")
            else:
                print(f"Failed to set secret '{key}'")
                sys.exit(1)
        
        elif command == "list":
            secrets = secrets_manager.list_secrets()
            print(f"Available secrets ({len(secrets)}):")
            for secret in secrets:
                print(f"  - {secret}")
        
        elif command == "stats":
            stats = secrets_manager.get_stats()
            print(json.dumps(stats, indent=2))
        
        elif command == "delete":
            if len(sys.argv) < 3:
                print("Usage: python secrets.py delete <key>")
                sys.exit(1)
            key = sys.argv[2]
            if secrets_manager.delete_secret(key):
                print(f"Secret '{key}' deleted")
            else:
                print(f"Failed to delete secret '{key}'")
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("AHGD Secrets Manager")
        print("Commands:")
        print("  get <key>        - Get secret value")
        print("  set <key> <val>  - Set secret value")
        print("  list             - List all secrets")
        print("  delete <key>     - Delete secret")
        print("  stats            - Show statistics")
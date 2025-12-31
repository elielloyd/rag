"""Authentication middleware for API key validation."""

from fastapi import HTTPException, Header, Depends
from cryptography.fernet import Fernet, InvalidToken
import base64
import hashlib

from config import settings


def _get_fernet_key(encryption_key: str) -> bytes:
    """
    Convert the encryption key to a valid Fernet key.
    Fernet requires a 32-byte base64-encoded key.
    """
    # Hash the encryption key to get a consistent 32-byte key
    key_hash = hashlib.sha256(encryption_key.encode()).digest()
    # Base64 encode it for Fernet
    return base64.urlsafe_b64encode(key_hash)


def decrypt_api_key(encrypted_value: str) -> str:
    """
    Decrypt the encrypted API key using the encryption key from settings.
    
    Args:
        encrypted_value: The encrypted API key from the x-api-key header
        
    Returns:
        The decrypted API key
        
    Raises:
        HTTPException: If decryption fails
    """
    if not settings.encryption_key:
        raise HTTPException(
            status_code=500,
            detail="Server encryption key not configured"
        )
    
    try:
        fernet_key = _get_fernet_key(settings.encryption_key)
        fernet = Fernet(fernet_key)
        decrypted = fernet.decrypt(encrypted_value.encode())
        return decrypted.decode()
    except InvalidToken:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key - decryption failed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format"
        )


def get_api_key_header(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """
    FastAPI dependency to extract the x-api-key header.
    
    Args:
        x_api_key: The API key from the request header
        
    Returns:
        The API key value
    """
    return x_api_key


def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> bool:
    """
    FastAPI dependency to verify the API key.
    
    This dependency:
    1. Extracts the x-api-key header
    2. Decrypts it using the ENCRYPTION_KEY from env
    3. Compares with the API_KEY from env
    4. Raises 401 if invalid
    
    Args:
        x_api_key: The encrypted API key from the request header
        
    Returns:
        True if the API key is valid
        
    Raises:
        HTTPException: If the API key is missing, invalid, or doesn't match
    """
    if not settings.api_key:
        raise HTTPException(
            status_code=500,
            detail="Server API key not configured"
        )
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing x-api-key header"
        )
    
    # Decrypt the incoming API key
    decrypted_key = decrypt_api_key(x_api_key)
    
    # Compare with the expected API key
    if decrypted_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True


def encrypt_api_key(api_key: str) -> str:
    """
    Utility function to encrypt an API key.
    This can be used to generate encrypted keys for clients.
    
    Args:
        api_key: The plain text API key to encrypt
        
    Returns:
        The encrypted API key
    """
    if not settings.encryption_key:
        raise ValueError("ENCRYPTION_KEY not configured")
    
    fernet_key = _get_fernet_key(settings.encryption_key)
    fernet = Fernet(fernet_key)
    encrypted = fernet.encrypt(api_key.encode())
    return encrypted.decode()

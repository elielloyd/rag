#!/usr/bin/env python3
"""
Script to generate encrypted API keys for client authentication.

This script encrypts an API key using the same encryption method as the server,
allowing you to generate valid x-api-key header values offline.

Usage:
    python scripts/generate_api_key.py --api-key "your-api-key" --encryption-key "your-encryption-key"
    
    Or with environment variables:
    export ENCRYPTION_KEY="your-encryption-key"
    python scripts/generate_api_key.py --api-key "your-api-key"
"""

import argparse
import base64
import hashlib
import os
import sys

from cryptography.fernet import Fernet


def get_fernet_key(encryption_key: str) -> bytes:
    """
    Convert the encryption key to a valid Fernet key.
    Fernet requires a 32-byte base64-encoded key.
    """
    key_hash = hashlib.sha256(encryption_key.encode()).digest()
    return base64.urlsafe_b64encode(key_hash)


def encrypt_api_key(api_key: str, encryption_key: str) -> str:
    """
    Encrypt an API key using Fernet encryption.
    
    Args:
        api_key: The plain text API key to encrypt
        encryption_key: The encryption key (shared secret)
        
    Returns:
        The encrypted API key (base64 encoded)
    """
    fernet_key = get_fernet_key(encryption_key)
    fernet = Fernet(fernet_key)
    encrypted = fernet.encrypt(api_key.encode())
    return encrypted.decode()


def decrypt_api_key(encrypted_key: str, encryption_key: str) -> str:
    """
    Decrypt an encrypted API key (for verification).
    
    Args:
        encrypted_key: The encrypted API key
        encryption_key: The encryption key (shared secret)
        
    Returns:
        The decrypted API key
    """
    fernet_key = get_fernet_key(encryption_key)
    fernet = Fernet(fernet_key)
    decrypted = fernet.decrypt(encrypted_key.encode())
    return decrypted.decode()


def main():
    parser = argparse.ArgumentParser(
        description="Generate encrypted API keys for TrueClaim API authentication"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="The API key to encrypt"
    )
    parser.add_argument(
        "--encryption-key",
        default=os.environ.get("ENCRYPTION_KEY"),
        help="The encryption key (or set ENCRYPTION_KEY env var)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the encrypted key by decrypting it"
    )
    
    args = parser.parse_args()
    
    if not args.encryption_key:
        print("Error: Encryption key is required.")
        print("Provide --encryption-key or set ENCRYPTION_KEY environment variable.")
        sys.exit(1)
    
    # Generate encrypted key
    encrypted_key = encrypt_api_key(args.api_key, args.encryption_key)
    
    print("\n" + "=" * 60)
    print("ENCRYPTED API KEY GENERATED")
    print("=" * 60)
    print(f"\nEncrypted Key:\n{encrypted_key}")
    print("\n" + "-" * 60)
    print("Usage: Add this value to the 'x-api-key' header in your requests")
    print("-" * 60)
    print("\nExample curl command:")
    print(f'curl -H "x-api-key: {encrypted_key}" ...')
    print("=" * 60 + "\n")
    
    # Verify if requested
    if args.verify:
        decrypted = decrypt_api_key(encrypted_key, args.encryption_key)
        if decrypted == args.api_key:
            print("✓ Verification successful: Decrypted key matches original")
        else:
            print("✗ Verification failed: Decrypted key does not match")
            sys.exit(1)


if __name__ == "__main__":
    main()

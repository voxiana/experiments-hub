#!/usr/bin/env python3
"""
Document Ingestion Script
Ingest documents into the RAG knowledge base from various sources
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

RAG_SERVICE_URL = "http://localhost:8080"
GATEWAY_URL = "http://localhost:8000"

# ============================================================================
# Ingestion Functions
# ============================================================================

async def get_auth_token(tenant_id: str, api_key: str) -> str:
    """Get JWT token for authentication"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GATEWAY_URL}/auth/token",
            data={
                "tenant_id": tenant_id,
                "api_key": api_key,
            }
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Authentication failed: {resp.status}")

            data = await resp.json()
            return data["access_token"]


async def ingest_file(
    file_path: str,
    tenant_id: str,
    token: str,
    title: str = None,
) -> dict:
    """Ingest a single file"""
    logger.info(f"Ingesting file: {file_path}")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    async with aiohttp.ClientSession() as session:
        # Prepare multipart form data
        data = aiohttp.FormData()
        data.add_field('tenant_id', tenant_id)

        if title:
            data.add_field('title', title)
        else:
            data.add_field('title', file_path.name)

        # Add file
        data.add_field(
            'file',
            open(file_path, 'rb'),
            filename=file_path.name,
            content_type='application/octet-stream',
        )

        # Post to ingestion endpoint
        async with session.post(
            f"{RAG_SERVICE_URL}/ingest",
            data=data,
            headers={'Authorization': f'Bearer {token}'},
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Ingestion failed: {resp.status} - {error}")

            result = await resp.json()
            logger.info(
                f"✅ Ingested {file_path.name}: "
                f"{result['chunks_created']} chunks in {result['took_ms']:.0f}ms"
            )

            return result


async def ingest_text(
    text: str,
    tenant_id: str,
    token: str,
    title: str = "Text Document",
) -> dict:
    """Ingest raw text"""
    logger.info(f"Ingesting text: {title}")

    async with aiohttp.ClientSession() as session:
        payload = {
            "tenant_id": tenant_id,
            "text": text,
            "title": title,
        }

        async with session.post(
            f"{RAG_SERVICE_URL}/ingest",
            json=payload,
            headers={'Authorization': f'Bearer {token}'},
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Ingestion failed: {resp.status} - {error}")

            result = await resp.json()
            logger.info(
                f"✅ Ingested text '{title}': "
                f"{result['chunks_created']} chunks"
            )

            return result


async def ingest_directory(
    directory: str,
    tenant_id: str,
    token: str,
    extensions: List[str] = None,
) -> dict:
    """Ingest all files in a directory"""
    if extensions is None:
        extensions = ['.pdf', '.docx', '.txt', '.md']

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Find all matching files
    files = []
    for ext in extensions:
        files.extend(dir_path.glob(f"**/*{ext}"))

    logger.info(f"Found {len(files)} files in {directory}")

    # Ingest files
    results = []
    for file_path in files:
        try:
            result = await ingest_file(
                str(file_path),
                tenant_id,
                token,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    return {
        "total_files": len(files),
        "success": len(results),
        "failed": len(files) - len(results),
        "results": results,
    }


async def ingest_sample_kb(tenant_id: str, token: str):
    """Ingest sample knowledge base (demo FAQs)"""
    logger.info("Ingesting sample knowledge base...")

    sample_docs = [
        {
            "title": "Support Hours - English",
            "text": """
            Our customer support is available:
            - Sunday to Thursday: 9 AM to 6 PM UAE Time (UTC+4)
            - Friday and Saturday: Closed

            During business hours, you can reach us via:
            - Phone: +971-4-123-4567
            - Email: support@example.ae
            - Live Chat on our website

            For urgent issues outside business hours, please send an email
            and we'll respond within 24 hours.
            """,
        },
        {
            "title": "Support Hours - Arabic",
            "text": """
            ساعات دعم العملاء:
            - الأحد إلى الخميس: 9 صباحاً - 6 مساءً (توقيت الإمارات)
            - الجمعة والسبت: مغلق

            خلال ساعات العمل، يمكنك التواصل معنا عبر:
            - الهاتف: 4567-123-4-971+
            - البريد الإلكتروني: support@example.ae
            - الدردشة المباشرة على موقعنا

            للمسائل العاجلة خارج ساعات العمل، يرجى إرسال بريد إلكتروني
            وسنرد خلال 24 ساعة.
            """,
        },
        {
            "title": "Shipping Policy",
            "text": """
            Shipping Information:

            Within UAE:
            - Free shipping on orders over AED 200
            - Standard delivery: 1-2 business days
            - Express delivery: Same day (if ordered before 12 PM)

            International Shipping:
            - GCC countries: 3-5 business days
            - Other countries: 7-14 business days

            Tracking:
            You will receive a tracking number via SMS and email once your
            order ships. Track your order at: https://example.ae/track

            Returns:
            14-day return policy. Item must be unused and in original packaging.
            """,
        },
        {
            "title": "Payment Methods",
            "text": """
            Accepted Payment Methods:

            1. Credit/Debit Cards:
               - Visa, Mastercard, American Express
               - Secure 3D authentication

            2. Digital Wallets:
               - Apple Pay, Google Pay, Samsung Pay

            3. Cash on Delivery (COD):
               - Available within UAE
               - AED 15 service fee

            4. Bank Transfer:
               - IBAN: AE07 0000 0000 0000 0000 001
               - Bank: Emirates NBD

            All transactions are encrypted and secure.
            """,
        },
        {
            "title": "Account & Profile",
            "text": """
            Managing Your Account:

            Create Account:
            Sign up at https://example.ae/register with your email or phone number.

            Profile Information:
            Update your profile, shipping addresses, and payment methods
            in Account Settings.

            Password Reset:
            Click "Forgot Password" on the login page. You'll receive a
            reset link via email.

            Newsletter:
            Manage email preferences in Account > Notifications.

            Delete Account:
            Contact support@example.ae to request account deletion.
            All data will be removed within 30 days.
            """,
        },
    ]

    results = []
    for doc in sample_docs:
        try:
            result = await ingest_text(
                text=doc["text"],
                tenant_id=tenant_id,
                token=token,
                title=doc["title"],
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to ingest {doc['title']}: {e}")

    logger.info(f"✅ Sample KB ingested: {len(results)} documents")

    return results


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Voice AI CX knowledge base"
    )
    parser.add_argument(
        '--tenant-id',
        default='demo',
        help='Tenant ID',
    )
    parser.add_argument(
        '--api-key',
        default='demo_key',
        help='API key for authentication',
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # File ingestion
    file_parser = subparsers.add_parser('file', help='Ingest a single file')
    file_parser.add_argument('path', help='Path to file')
    file_parser.add_argument('--title', help='Document title')

    # Directory ingestion
    dir_parser = subparsers.add_parser('directory', help='Ingest directory')
    dir_parser.add_argument('path', help='Path to directory')
    dir_parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.pdf', '.docx', '.txt', '.md'],
        help='File extensions to ingest',
    )

    # Text ingestion
    text_parser = subparsers.add_parser('text', help='Ingest raw text')
    text_parser.add_argument('text', help='Text content')
    text_parser.add_argument('--title', required=True, help='Document title')

    # Sample KB
    subparsers.add_parser('sample', help='Ingest sample knowledge base')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Authenticate
        logger.info(f"Authenticating as tenant: {args.tenant_id}")
        token = await get_auth_token(args.tenant_id, args.api_key)

        # Execute command
        if args.command == 'file':
            await ingest_file(args.path, args.tenant_id, token, args.title)

        elif args.command == 'directory':
            result = await ingest_directory(
                args.path,
                args.tenant_id,
                token,
                args.extensions,
            )
            logger.info(
                f"✅ Directory ingestion complete: "
                f"{result['success']}/{result['total_files']} files"
            )

        elif args.command == 'text':
            await ingest_text(args.text, args.tenant_id, token, args.title)

        elif args.command == 'sample':
            await ingest_sample_kb(args.tenant_id, token)

        logger.info("✅ Ingestion complete!")

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())

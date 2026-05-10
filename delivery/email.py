import base64
import json
import os
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise KeyError(f"Required environment variable '{name}' is not set")
    return value


def _html_to_text(html: str) -> str:
    """Convert an HTML fragment to a readable plain-text equivalent.

    Headings become ALL-CAPS lines, list items become '- ' prefixed lines,
    and hyperlinks become 'anchor text (url)' inline.
    """
    # Resolve links before stripping tags: <a href="url">text</a> → text (url)
    text = re.sub(
        r'<a\s[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        lambda m: f'{re.sub(r"<[^>]+>", "", m.group(2)).strip()} ({m.group(1)})',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Headings → ALL-CAPS lines with surrounding whitespace
    def _as_heading(m: re.Match) -> str:
        content = re.sub(r'<[^>]+>', '', m.group(2)).strip().upper()
        return f'\n\n{content}\n'

    text = re.sub(
        r'<(h[1-6])[^>]*>(.*?)</\1>',
        _as_heading,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # List items → "- item"
    text = re.sub(
        r'<li[^>]*>(.*?)</li>',
        lambda m: f'- {re.sub(r"<[^>]+>", "", m.group(1)).strip()}\n',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Block-level endings → line breaks
    text = re.sub(r'</p>|<br\s*/?>|</div>|</ul>|</ol>', '\n', text, flags=re.IGNORECASE)

    # Strip remaining tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalise consecutive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _decode_jwt_claims(token: str) -> dict:
    """Decode the JWT payload without signature verification.
    Used only to log token roles for CI diagnostics.
    """
    parts = token.split('.')
    if len(parts) < 2:
        return {}
    padding = '=' * (-len(parts[1]) % 4)
    try:
        decoded = base64.urlsafe_b64decode(parts[1] + padding)
        return json.loads(decoded.decode('utf-8'))
    except Exception:
        return {}


def get_access_token() -> str:
    """Acquire a client-credentials OAuth 2.0 access token from Azure AD."""
    tenant_id = _require_env('MS_TENANT_ID')
    client_id = _require_env('MS_CLIENT_ID')
    client_secret = _require_env('MS_CLIENT_SECRET')

    url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
    response = requests.post(
        url,
        data={
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'https://graph.microsoft.com/.default',
            'grant_type': 'client_credentials',
        },
        timeout=30,
    )
    response.raise_for_status()

    token = response.json()['access_token']
    roles = _decode_jwt_claims(token).get('roles', [])
    print(f'[Email] Token acquired. Roles: {roles}')

    return token


def send_email_digest(subject: str, html_body: str) -> None:
    """Send the digest as a multipart/alternative email via Microsoft Graph.

    Wraps the HTML fragment in a full document envelope and derives a
    plain-text fallback for clients that do not render HTML.
    Raises RuntimeError on any Graph API failure.
    """
    sender = _require_env('MS_SENDER_EMAIL')
    recipient = os.environ.get('DIGEST_RECIPIENT') or sender

    full_html = f'<html><body>\n{html_body}\n</body></html>'
    plain_text = _html_to_text(html_body)

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    msg.attach(MIMEText(full_html, 'html', 'utf-8'))

    token = get_access_token()

    # Graph MIME submission: base64-encode the raw MIME message.
    # Content-Type: text/plain signals to Graph that the body is a MIME string.
    url = f'https://graph.microsoft.com/v1.0/users/{sender}/sendMail'
    mime_b64 = base64.b64encode(msg.as_bytes()).decode('ascii')

    response = requests.post(
        url,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'text/plain',
        },
        data=mime_b64,
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f'Graph sendMail failed: {response.status_code} {response.text}'
        )

    print(f'[Email] Digest sent to {recipient}.')

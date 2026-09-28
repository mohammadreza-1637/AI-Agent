import os
import re
import sys
import json
import logging
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------
# Paths (works for .py and .exe)
# --------------------------------

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Max size for a single download (bytes). 500 MB by default.
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024

load_dotenv(BASE_DIR / ".env")


# --------------------------------
# Logging to file
# --------------------------------

log_file = DOWNLOADS_DIR / "agent.log"

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if the module reloads
if not logger.handlers:
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(fh)


# --------------------------------
# Config
# --------------------------------

api_key = os.getenv("AI_API_KEY")
base_url = os.getenv("AI_BASE_URL")
model = os.getenv("MODEL_NAME")

if not api_key:
    raise ValueError("AI_API_KEY is not set (check your .env file)")
if not base_url:
    raise ValueError("AI_BASE_URL is not set (check your .env file)")
if not model:
    raise ValueError("MODEL_NAME is not set (check your .env file)")

if not base_url.startswith(("http://", "https://")):
    raise ValueError(
        f"AI_BASE_URL must start with http:// or https:// — got: {base_url!r}"
    )

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=180.0,
    max_retries=1,
)


# --------------------------------
# Shared HTTP session
# --------------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
    "Connection": "keep-alive",
})


# --------------------------------
# Helpers
# --------------------------------

DOWNLOADABLE_EXTS = (
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm", ".apk",
    ".srt", ".ass", ".sub",
    ".pdf", ".epub",
)


def _is_file_url(url):
    """True if the URL path ends with a known downloadable extension."""
    path = url.lower().split("?")[0].split("#")[0]
    return path.endswith(DOWNLOADABLE_EXTS)


def _filename_from_response(r, fallback_url):
    cd = r.headers.get("Content-Disposition", "") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        return m.group(1)
    name = fallback_url.split("?")[0].rstrip("/").split("/")[-1]
    return name or "download.bin"


def _sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|\r\n\t]', "_", name)
    return name.strip() or "download.bin"


# --------------------------------
# Tools
# --------------------------------

def get_website(url):
    """Fetch a page's HTML (up to 200k chars)."""
    logger.info(f"get_website url={url}")
    try:
        r = session.get(url, timeout=(10, 30))
        r.raise_for_status()
        logger.info(f"get_website ok status={r.status_code} url={url}")
        return {
            "success": True,
            "url": url,
            "final_url": str(r.url),
            "status_code": r.status_code,
            "content_type": r.headers.get("Content-Type", ""),
            "content": r.text[:200000],
            "truncated": len(r.text) > 200000,
        }
    except requests.RequestException as e:
        logger.warning(f"get_website failed url={url} error={e}")
        return {
            "success": False,
            "url": url,
            "status_code": getattr(e.response, "status_code", None),
            "error": str(e),
        }


def find_download_links(page_url):
    """
    Scan a page (full HTML, no truncation) and return only direct file URLs.
    """
    logger.info(f"find_download_links url={page_url}")
    try:
        r = session.get(page_url, timeout=(10, 30))
        r.raise_for_status()

        final_url = str(r.url)
        html = r.text

        candidates = set()

        for attr in ("href", "src", "data-href", "data-url", "data-download"):
            for m in re.finditer(
                rf'{attr}\s*=\s*["\']([^"\']+)["\']',
                html,
                flags=re.IGNORECASE,
            ):
                candidates.add(m.group(1))

        for m in re.finditer(
            r'https?://[^\s"\'<>)]+',
            html,
            flags=re.IGNORECASE,
        ):
            candidates.add(m.group(0))

        absolute = set()
        for c in candidates:
            if c.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            try:
                absolute.add(urljoin(final_url, c))
            except Exception:
                continue

        file_links = [u for u in absolute if _is_file_url(u)]

        seen = set()
        unique = []
        for u in file_links:
            if u not in seen:
                seen.add(u)
                unique.append(u)

        logger.info(
            f"find_download_links found {len(unique)} file link(s) "
            f"on {page_url}"
        )

        return {
            "success": True,
            "page_url": page_url,
            "final_url": final_url,
            "count": len(unique),
            "file_links": unique[:200],
        }

    except requests.RequestException as e:
        logger.warning(f"find_download_links failed url={page_url} error={e}")
        return {
            "success": False,
            "page_url": page_url,
            "error": str(e),
        }


def download_file(url, filename=None):
    """
    Download a file from a direct URL and save it to downloads/.
    Referer is set dynamically from the file's own origin.
    Enforces a max size limit (MAX_DOWNLOAD_BYTES).
    """
    if not _is_file_url(url):
        logger.warning(f"download_file rejected non-file URL: {url}")
        print(f"[download] REJECTED (not a file URL): {url}")
        return {
            "success": False,
            "url": url,
            "error": (
                "This URL does not point to a downloadable file "
                "(it does not end with .zip, .rar, .7z, .exe, .msi, etc.). "
                "Find a real file link in the page HTML first."
            ),
        }

    try:
        parsed = urlparse(url)
        referer = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme else None

        headers = {}
        if referer:
            headers["Referer"] = referer

        print(f"[download] GET {url}")
        logger.info(f"download_file url={url}")

        with session.get(
            url,
            stream=True,
            timeout=(15, 300),
            headers=headers,
            allow_redirects=True,
        ) as r:
            status = r.status_code
            ctype = r.headers.get("Content-Type")
            clen = r.headers.get("Content-Length")

            print(
                f"[download] status={status}, "
                f"content-type={ctype}, "
                f"content-length={clen}"
            )
            logger.info(
                f"download_file status={status} type={ctype} length={clen}"
            )

            r.raise_for_status()

            # Size limit check
            if clen is not None:
                try:
                    size = int(clen)
                    if size > MAX_DOWNLOAD_BYTES:
                        logger.warning(
                            f"download_file size {size} exceeds limit "
                            f"{MAX_DOWNLOAD_BYTES}"
                        )
                        return {
                            "success": False,
                            "url": url,
                            "error": (
                                f"File size ({size} bytes) exceeds the "
                                f"limit of {MAX_DOWNLOAD_BYTES} bytes. "
                                "Refusing to download."
                            ),
                        }
                except ValueError:
                    pass

            if not filename:
                filename = _filename_from_response(r, str(r.url) or url)

            filename = _sanitize_filename(filename)

            # Refuse to save HTML pages
            if filename.lower().endswith((".html", ".htm", ".php")):
                logger.warning(
                    f"download_file server returned HTML, filename={filename}"
                )
                return {
                    "success": False,
                    "url": url,
                    "error": (
                        "Server returned an HTML page instead of a file. "
                        "The link is probably wrong or requires a session."
                    ),
                }

            filepath = DOWNLOADS_DIR / filename

            if filepath.exists():
                logger.info(f"download_file skipped (exists) file={filename}")
                return {
                    "success": True,
                    "already_exists": True,
                    "filename": filename,
                    "saved_to": str(filepath.resolve()),
                    "size_bytes": filepath.stat().st_size,
                }

            total = 0
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)

            print(f"[download] saved {total} bytes to {filepath}")
            logger.info(f"download_file saved {total} bytes to {filepath}")

            return {
                "success": True,
                "url": url,
                "final_url": str(r.url),
                "filename": filename,
                "size_bytes": filepath.stat().st_size,
                "saved_to": str(filepath.resolve()),
            }

    except requests.RequestException as e:
        print(f"[download] FAILED: {e}")
        logger.warning(f"download_file failed url={url} error={e}")
        return {"success": False, "url": url, "error": str(e)}


def extract_zip(filename):
    """
    Extract a downloaded .zip file into a folder next to it.
    RAR and 7z are NOT supported here (they need extra tools).
    """
    logger.info(f"extract_zip filename={filename}")
    filepath = DOWNLOADS_DIR / filename

    if not filepath.exists():
        return {
            "success": False,
            "error": f"File not found: {filepath}",
        }

    if filepath.suffix.lower() != ".zip":
        return {
            "success": False,
            "error": (
                "Only .zip archives can be extracted here. "
                f"Got: {filepath.suffix}"
            ),
        }

    target_dir = DOWNLOADS_DIR / filepath.stem

    try:
        target_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(target_dir)

        extracted = [p.name for p in target_dir.iterdir()]
        logger.info(
            f"extract_zip ok file={filename} -> {target_dir} "
            f"({len(extracted)} items)"
        )
        return {
            "success": True,
            "zip_path": str(filepath.resolve()),
            "extracted_to": str(target_dir.resolve()),
            "items": extracted[:50],
            "count": len(extracted),
        }
    except zipfile.BadZipFile:
        logger.warning(f"extract_zip bad zip: {filename}")
        return {
            "success": False,
            "error": "This file is not a valid ZIP archive.",
        }
    except Exception as e:
        logger.warning(f"extract_zip failed file={filename} error={e}")
        return {"success": False, "error": str(e)}


# --------------------------------
# Tool schemas
# --------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_website",
            "description": (
                "Fetch the HTML of any web page. Use this to explore a "
                "site, read a page, or look at search results. Do NOT use "
                "this to visit archive.org or web.archive.org."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_download_links",
            "description": (
                "Scan a page and return ONLY real direct-file URLs "
                "(.zip, .rar, .7z, .exe, .msi, .dmg, .apk, .srt, .pdf, "
                "etc.) found anywhere in the page HTML. Use this before "
                "download_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_url": {
                        "type": "string",
                        "description": "The page URL to scan for file links.",
                    }
                },
                "required": ["page_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_file",
            "description": (
                "Download a file from a DIRECT FILE URL and save it to the "
                "local downloads/ folder. The URL MUST end with a real file "
                "extension (.zip, .rar, .7z, .exe, .msi, .dmg, .apk, .srt, "
                ".pdf, ...). NEVER pass a web page URL. Files larger than "
                "the configured limit are rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct file URL."},
                    "filename": {
                        "type": "string",
                        "description": "Optional filename to save as.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_zip",
            "description": (
                "Extract a previously downloaded .zip file into a folder "
                "next to it (inside downloads/). Only works for .zip — "
                "not .rar or .7z. Use this when the user asks to unzip "
                "or extract a downloaded archive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "The filename of the downloaded .zip "
                            "(just the name, not the full path)."
                        ),
                    }
                },
                "required": ["filename"],
            },
        },
    },
]


# --------------------------------
# System prompt
# --------------------------------

SYSTEM_PROMPT = (
    "You are a general-purpose web research and download agent. "
    "You can work with ANY website the user names.\n\n"
    "Available tools:\n"
    "- get_website(url): fetch a page's HTML.\n"
    "- find_download_links(page_url): scan a page and return only real "
    "  direct-file URLs (.zip, .rar, .exe, ...).\n"
    "- download_file(url): download a file from a direct URL to the "
    "  local downloads/ folder.\n"
    "- extract_zip(filename): extract a downloaded .zip archive "
    "  (only .zip, not .rar or .7z).\n\n"
    "General workflow:\n"
    "1. If the user gives a homepage, use get_website(homepage) to find "
    "   a search box or the relevant page. If the site supports a search "
    "   URL pattern (e.g. `?s=<query>`), go straight there.\n"
    "2. Once on the item's detail page, call find_download_links(page_url) "
    "   to get the real file URLs.\n"
    "3. Pick the best candidate that matches the user's request.\n"
    "4. Call download_file(url) with that URL.\n"
    "5. If the user asked to extract/unzip AND the file is a .zip, call "
    "   extract_zip(filename).\n"
    "6. Report the saved filename, size, full local path, and (if "
    "   extracted) the folder.\n\n"
    "NEVER GUESS URLs — the most important rule:\n"
    "- download_file may ONLY be called with a URL that was literally "
    "  returned by find_download_links or appeared verbatim in HTML you "
    "  fetched with get_website.\n"
    "- Do NOT build a URL from a pattern. If the URL isn't in your tool "
    "  results, you cannot use it.\n"
    "- A valid URL for download_file MUST end with a real file extension.\n"
    "- Never pass a web page URL to download_file.\n"
    "- If find_download_links returns nothing, open the page with "
    "  get_website, look for file links, and use those exact URLs. If "
    "  you still can't find any, tell the user instead of guessing.\n\n"
    "STOPPING RULES:\n"
    "- After download_file returns success=true, if the user did NOT ask "
    "  to extract, STOP and report the result. Do not call more tools.\n"
    "- If the user DID ask to extract and the file is .zip, call "
    "  extract_zip once, then STOP.\n"
    "- If download_file returns success=false, STOP. Do NOT try "
    "  archive.org or alternate sites. Explain the error and suggest "
    "  alternatives.\n"
    "- Never visit archive.org or web.archive.org unless explicitly asked.\n"
    "- Do not retry a failed download more than once.\n\n"
    "Other rules:\n"
    "- Always prefer https:// over http:// when both exist.\n"
    "- If a page is a link-shortener or countdown, follow it with "
    "  get_website to reach the real file link.\n"
    "- If a site requires login/payment, tell the user clearly.\n"
    "- Answer in the same language the user wrote in."
)


# --------------------------------
# Agent loop
# --------------------------------

def ask_agent(user_message):
    logger.info(f"USER: {user_message}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_iterations = 15
    successful_downloads = []
    successful_extractions = []

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            if msg.content:
                logger.info(f"AGENT: {msg.content}")
            return msg.content

        download_failed = False
        extraction_failed = False

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "get_website":
                url = args.get("url", "")
                print(f"\n[Tool] get_website({url})")
                result = get_website(url)

            elif name == "find_download_links":
                page_url = args.get("page_url", "")
                print(f"\n[Tool] find_download_links({page_url})")
                result = find_download_links(page_url)

            elif name == "download_file":
                url = args.get("url", "")
                filename = args.get("filename")
                print(f"\n[Tool] download_file({url})")
                result = download_file(url, filename)

                if isinstance(result, dict):
                    if result.get("success"):
                        successful_downloads.append(result)
                    else:
                        download_failed = True

            elif name == "extract_zip":
                filename = args.get("filename", "")
                print(f"\n[Tool] extract_zip({filename})")
                result = extract_zip(filename)

                if isinstance(result, dict):
                    if result.get("success"):
                        successful_extractions.append(result)
                    else:
                        extraction_failed = True

            else:
                result = {"success": False, "error": f"Unknown tool: {name}"}

            short = json.dumps(result, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": short,
            })

        # If download succeeded and user asked to extract, let the loop
        # continue so the model can call extract_zip.
        if successful_extractions:
            d = successful_downloads[0] if successful_downloads else None
            e = successful_extractions[0]
            lines = ["✅ Done:"]
            if d:
                lines.append(f"   File      : {d['filename']}")
                lines.append(f"   Size      : {d['size_bytes']:,} bytes")
                lines.append(f"   Path      : {d['saved_to']}")
            lines.append(f"   Extracted : {e['extracted_to']}")
            lines.append(f"   Items     : {e.get('count', 0)}")
            return "\n".join(lines)

        if successful_downloads:
            d = successful_downloads[0]
            # If the file is a .zip, leave the model one more chance to
            # call extract_zip if the user asked for it. Otherwise finish.
            # We finish here and let the user re-ask to extract if wanted.
            return (
                f"✅ Downloaded successfully:\n"
                f"   File : {d['filename']}\n"
                f"   Size : {d['size_bytes']:,} bytes\n"
                f"   Path : {d['saved_to']}"
            )

        if download_failed or extraction_failed:
            messages.append({
                "role": "user",
                "content": (
                    "The last operation FAILED. Do NOT call any more tools "
                    "and do NOT try archive.org or any other site. In the "
                    "user's language, explain the error and suggest "
                    "alternatives. Do not guess URLs."
                ),
            })
            final = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content = final.choices[0].message.content
            if content:
                logger.info(f"AGENT: {content}")
            return content

    logger.warning("ask_agent stopped after max_iterations")
    return "Stopped after too many steps without finishing. Try rephrasing."


# --------------------------------
# Main loop
# --------------------------------

def main():
    print("AI Agent is ready. Type 'exit' to quit.\n")
    print(f"[config] downloads dir: {DOWNLOADS_DIR}")
    print(f"[config] max download:  {MAX_DOWNLOAD_BYTES:,} bytes")
    print(f"[config] log file:      {log_file}")
    print(f"[config] LLM: {base_url}  model={model}\n")

    logger.info("agent started")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info("agent stopped (interrupt)")
            break

        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            logger.info("agent stopped (exit)")
            break
        if not user_input.strip():
            continue

        try:
            answer = ask_agent(user_input)
            print(f"\nAgent: {answer}\n")
        except Exception as e:
            logger.exception(f"ask_agent error: {e}")
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
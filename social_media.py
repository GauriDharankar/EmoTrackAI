from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
import os
import re
from typing import Any
from urllib.parse import quote

import requests


USER_AGENT = "EmoTrack/1.0"
DEFAULT_TIMEOUT = 20


class SocialFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SocialPost:
    platform: str
    author: str
    text: str
    created_at: str | None
    url: str | None
    post_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlatformDescriptor:
    key: str
    label: str
    handle_help: str
    placeholder: str
    auth_hint: str
    needs_instance: bool = False
    instance_placeholder: str = ""


PLATFORM_DESCRIPTORS = {
    "bluesky": PlatformDescriptor(
        key="bluesky",
        label="Bluesky",
        handle_help="Public read access. Enter a Bluesky handle such as `alice.bsky.social`.",
        placeholder="alice.bsky.social",
        auth_hint="No token required for public posts.",
    ),
    "mastodon": PlatformDescriptor(
        key="mastodon",
        label="Mastodon",
        handle_help=(
            "Public read access. Enter `@user@server.tld` or a username plus the instance URL."
        ),
        placeholder="@gargron@mastodon.social",
        auth_hint="No token required for public posts on open instances.",
        needs_instance=True,
        instance_placeholder="https://mastodon.social",
    ),
    "x": PlatformDescriptor(
        key="x",
        label="X",
        handle_help="Requires an X API bearer token. Enter a handle such as `XDevelopers`.",
        placeholder="XDevelopers",
        auth_hint="Set `X_BEARER_TOKEN` in your environment or Streamlit secrets.",
    ),
}


def get_platform_descriptors():
    return list(PLATFORM_DESCRIPTORS.values())


def get_platform_descriptor(platform: str) -> PlatformDescriptor:
    try:
        return PLATFORM_DESCRIPTORS[platform]
    except KeyError as exc:
        raise SocialFetchError(f"Unsupported social platform: {platform}") from exc


def fetch_recent_posts(
    platform: str,
    handle: str,
    limit: int,
    instance_url: str | None = None,
    auth: dict[str, str] | None = None,
    fetch_all: bool = False,
    max_posts: int | None = None,
) -> list[SocialPost]:
    platform = platform.strip().lower()
    auth = auth or {}

    if platform == "bluesky":
        return BlueskyFetcher().fetch_posts(handle, limit, fetch_all=fetch_all, max_posts=max_posts)
    if platform == "mastodon":
        return MastodonFetcher().fetch_posts(
            handle,
            limit,
            instance_url=instance_url,
            fetch_all=fetch_all,
            max_posts=max_posts,
        )
    if platform == "x":
        return XFetcher(auth=auth).fetch_posts(
            handle,
            limit,
            fetch_all=fetch_all,
            max_posts=max_posts,
        )

    raise SocialFetchError(f"Unsupported social platform: {platform}")


def post_dicts(posts: list[SocialPost]) -> list[dict[str, Any]]:
    return [post.to_dict() for post in posts]


def _clean_text(text: str | None) -> str:
    text = text or ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _make_session(headers: dict[str, str] | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if headers:
        session.headers.update(headers)
    return session


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    try:
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = _extract_error_detail(exc.response)
        raise SocialFetchError(detail) from exc
    except requests.RequestException as exc:
        raise SocialFetchError(f"Network request failed: {exc}") from exc


def _extract_error_detail(response: requests.Response | None) -> str:
    if response is None:
        return "The social media request failed."

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("detail", "error", "message", "title"):
            value = payload.get(key)
            if value:
                return f"{response.status_code}: {value}"

        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                for key in ("detail", "message", "title"):
                    value = first.get(key)
                    if value:
                        return f"{response.status_code}: {value}"

    text = (response.text or "").strip()
    if text:
        return f"{response.status_code}: {text[:240]}"
    return f"{response.status_code}: The social media request failed."


def _ensure_text_posts(posts: list[SocialPost], platform_label: str) -> list[SocialPost]:
    filtered = [post for post in posts if post.text]
    if filtered:
        return filtered
    raise SocialFetchError(f"No text posts were returned from {platform_label}.")


class BlueskyFetcher:
    resolve_url = "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle"
    author_feed_url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"

    def fetch_posts(
        self,
        handle: str,
        limit: int,
        *,
        fetch_all: bool = False,
        max_posts: int | None = None,
    ) -> list[SocialPost]:
        normalized_handle = handle.strip().lstrip("@")
        if "." not in normalized_handle:
            normalized_handle = f"{normalized_handle}.bsky.social"

        session = _make_session()
        identity = _get_json(session, self.resolve_url, params={"handle": normalized_handle})
        did = identity.get("did")
        if not did:
            raise SocialFetchError("Bluesky handle could not be resolved.")

        posts: list[SocialPost] = []
        seen_post_ids: set[str] = set()
        target_posts = max_posts if fetch_all and max_posts else None
        cursor = None

        while True:
            page_goal = limit if not fetch_all else (target_posts - len(posts) if target_posts else 100)
            page_size = min(100, max(1, page_goal))
            payload = _get_json(
                session,
                self.author_feed_url,
                params={
                    "actor": did,
                    "filter": "posts_no_replies",
                    "limit": page_size,
                    **({"cursor": cursor} if cursor else {}),
                },
            )

            page_posts = payload.get("feed", [])
            if not page_posts:
                break

            for item in page_posts:
                post = item.get("post", {})
                record = post.get("record", {})
                author = post.get("author", {})
                text = _clean_text(record.get("text"))
                if not text:
                    continue

                uri = post.get("uri", "")
                rkey = uri.rsplit("/", 1)[-1] if uri else None
                post_id = post.get("cid") or rkey
                if post_id and post_id in seen_post_ids:
                    continue
                if post_id:
                    seen_post_ids.add(post_id)
                author_handle = author.get("handle", normalized_handle)
                post_url = (
                    f"https://bsky.app/profile/{author_handle}/post/{rkey}"
                    if rkey
                    else None
                )

                posts.append(
                    SocialPost(
                        platform="Bluesky",
                        author=author_handle,
                        text=text,
                        created_at=record.get("createdAt"),
                        url=post_url,
                        post_id=post_id,
                    )
                )

                if not fetch_all and len(posts) >= limit:
                    return _ensure_text_posts(posts[:limit], "Bluesky")
                if fetch_all and target_posts and len(posts) >= target_posts:
                    return _ensure_text_posts(posts[:target_posts], "Bluesky")

            cursor = payload.get("cursor")
            if not fetch_all or not cursor:
                break

        final_posts = posts if fetch_all and not target_posts else posts[: (target_posts or limit)]
        return _ensure_text_posts(final_posts, "Bluesky")


class MastodonFetcher:
    def fetch_posts(
        self,
        handle: str,
        limit: int,
        *,
        instance_url: str | None = None,
        fetch_all: bool = False,
        max_posts: int | None = None,
    ) -> list[SocialPost]:
        account_name, full_acct, instance = self._normalize_input(handle, instance_url)
        session = _make_session({"Accept": "application/json"})

        account = self._lookup_account(session, instance, account_name, full_acct)
        posts: list[SocialPost] = []
        seen_post_ids: set[str] = set()
        target_posts = max_posts if fetch_all and max_posts else None
        max_id = None

        while True:
            page_goal = limit if not fetch_all else (target_posts - len(posts) if target_posts else 80)
            page_size = max(1, min(page_goal, 80))
            payload = _get_json(
                session,
                f"{instance}/api/v1/accounts/{account['id']}/statuses",
                params={
                    "limit": page_size,
                    "exclude_replies": "true",
                    "exclude_reblogs": "true",
                    **({"max_id": max_id} if max_id else {}),
                },
            )

            if not payload:
                break

            for status in payload:
                text = _clean_text(status.get("content"))
                if not text:
                    continue

                post_id = str(status.get("id")) if status.get("id") is not None else None
                if post_id and post_id in seen_post_ids:
                    continue
                if post_id:
                    seen_post_ids.add(post_id)

                posts.append(
                    SocialPost(
                        platform="Mastodon",
                        author=account.get("acct") or account_name,
                        text=text,
                        created_at=status.get("created_at"),
                        url=status.get("url"),
                        post_id=post_id,
                    )
                )

                if not fetch_all and len(posts) >= limit:
                    return _ensure_text_posts(posts[:limit], "Mastodon")
                if fetch_all and target_posts and len(posts) >= target_posts:
                    return _ensure_text_posts(posts[:target_posts], "Mastodon")

            max_id = str(payload[-1].get("id")) if payload[-1].get("id") is not None else None
            if not fetch_all or not max_id:
                break

        final_posts = posts if fetch_all and not target_posts else posts[: (target_posts or limit)]
        return _ensure_text_posts(final_posts, "Mastodon")

    def _normalize_input(self, handle: str, instance_url: str | None) -> tuple[str, str, str]:
        normalized = handle.strip()
        normalized = normalized[1:] if normalized.startswith("@") else normalized

        if "@" in normalized:
            account_name, domain = normalized.split("@", 1)
            instance = instance_url.strip() if instance_url else f"https://{domain}"
            full_acct = f"{account_name}@{domain}"
        else:
            account_name = normalized
            full_acct = account_name
            if not instance_url:
                raise SocialFetchError(
                    "For Mastodon, enter a full handle like `@user@mastodon.social` "
                    "or provide the instance URL."
                )
            instance = instance_url.strip()

        if not account_name:
            raise SocialFetchError("Please enter a Mastodon account handle.")

        if not instance.startswith(("http://", "https://")):
            instance = f"https://{instance}"

        return account_name, full_acct.lower(), instance.rstrip("/")

    def _lookup_account(
        self,
        session: requests.Session,
        instance: str,
        account_name: str,
        full_acct: str,
    ) -> dict[str, Any]:
        candidates = []
        for query in (full_acct, account_name):
            payload = _get_json(
                session,
                f"{instance}/api/v2/search",
                params={"q": query, "type": "accounts", "limit": 5},
            )
            candidates.extend(payload.get("accounts", []))

        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            unique_candidates.append(candidate)

        exact_matches = [
            candidate
            for candidate in unique_candidates
            if candidate.get("acct", "").lower() == full_acct
            or candidate.get("username", "").lower() == account_name.lower()
        ]

        if exact_matches:
            return exact_matches[0]
        if unique_candidates:
            return unique_candidates[0]

        raise SocialFetchError(
            "Mastodon account was not found on that instance. "
            "Try the full handle and make sure the instance URL is correct."
        )


class XFetcher:
    base_url = "https://api.x.com/2"

    def __init__(self, auth: dict[str, str] | None = None):
        auth = auth or {}
        self.bearer_token = auth.get("x_bearer_token") or os.getenv("X_BEARER_TOKEN")

    def fetch_posts(
        self,
        handle: str,
        limit: int,
        *,
        fetch_all: bool = False,
        max_posts: int | None = None,
    ) -> list[SocialPost]:
        if not self.bearer_token:
            raise SocialFetchError(
                "X integration requires `X_BEARER_TOKEN`. "
                "Add it to your environment or Streamlit secrets."
            )

        username = handle.strip().lstrip("@")
        if not username:
            raise SocialFetchError("Please enter an X handle.")

        session = _make_session({"Authorization": f"Bearer {self.bearer_token}"})

        user = _get_json(
            session,
            f"{self.base_url}/users/by/username/{quote(username)}",
            params={"user.fields": "id,name,username"},
        ).get("data")
        if not user:
            raise SocialFetchError("X account was not found.")

        posts: list[SocialPost] = []
        seen_post_ids: set[str] = set()
        target_posts = max_posts if fetch_all and max_posts else None
        pagination_token = None

        while True:
            page_goal = limit if not fetch_all else (target_posts - len(posts) if target_posts else 100)
            page_size = max(5, min(page_goal, 100))
            payload = _get_json(
                session,
                f"{self.base_url}/users/{user['id']}/tweets",
                params={
                    "max_results": page_size,
                    "exclude": "replies,retweets",
                    "tweet.fields": "created_at,lang,public_metrics",
                    **({"pagination_token": pagination_token} if pagination_token else {}),
                },
            )

            page_items = payload.get("data", [])
            if not page_items:
                break

            for item in page_items:
                text = _clean_text(item.get("text"))
                if not text:
                    continue

                tweet_id = str(item.get("id")) if item.get("id") is not None else None
                if tweet_id and tweet_id in seen_post_ids:
                    continue
                if tweet_id:
                    seen_post_ids.add(tweet_id)
                post_url = f"https://x.com/{user['username']}/status/{tweet_id}" if tweet_id else None
                posts.append(
                    SocialPost(
                        platform="X",
                        author=user["username"],
                        text=text,
                        created_at=item.get("created_at"),
                        url=post_url,
                        post_id=tweet_id,
                    )
                )

                if not fetch_all and len(posts) >= limit:
                    return _ensure_text_posts(posts[:limit], "X")
                if fetch_all and target_posts and len(posts) >= target_posts:
                    return _ensure_text_posts(posts[:target_posts], "X")

            pagination_token = payload.get("meta", {}).get("next_token")
            if not fetch_all or not pagination_token:
                break

        final_posts = posts if fetch_all and not target_posts else posts[: (target_posts or limit)]
        return _ensure_text_posts(final_posts, "X")
